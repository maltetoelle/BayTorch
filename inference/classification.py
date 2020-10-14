import torch
import time
import sys
import numpy as np

class ClassificationTrainer(object):
    def __init__(self,
                net,
                train_loader,
                val_loader=None,
                device_ids=None):

        super(Trainer, self).__init__()

        self.device_ids = device_ids
        use_cuda = torch.cuda.is_available() and len(self.device_ids) > 0
        self.device = torch.device("cuda:{}".format(device_ids[0]) if use_cuda else "cpu")
        if len(self.device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
        self.net = net.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, n_epochs, criterion, optimizer, scheduler=None, verbosity=1,
              opt_kwargs=dict(), crit_kwargs=dict(), sched_kwargs=dict()):

        criterion = criterion(reduction='sum', **crit_kwargs)
        optimizer = optimizer(filter(lambda p: p.requires_grad, self.net.parameters()), **opt_kwargs)
        scheduler = scheduler(optimizer, **sched_kwargs)

        nll = []
        kl = []
        accuracy_train = []
        accuracy_test = []

        n_minibatches = np.ceil(float(len(self.train_loader.dataset)) / self.train_loader.batch_size)
        beta = torch.tensor(1.0/(n_minibatches))

        start = time.time()
        for epoch in range(n_epochs):

            info_dict = self.train_epoch(criterion, optimizer, beta, epoch)
            if self.val_loader is not None:
                accuracy = self.evaluate(self.val_loader)
                accuracy_test.append(accuracy)
            scheduler.step()

            nll.append(info_dict["nll"])
            kl.append(info_dict["kl"])
            accuracy_train.append(info_dict["accuracy"])

            if verbosity and (epoch+1) % verbosity == 0:
                if self.val_loader is not None:
                    print("#{:4d} | ELBO Loss: {:7.2f} | Accuracy: {:6.2f} % [{:6.2f} %] | KL: {:7.2f} | NLL: {:7.2f} |"\
                          .format(epoch+1, np.sum(info_dict["nll"]) + np.sum(info_dict["kl"]), info_dict["accuracy"], \
                                  accuracy, np.sum(info_dict["kl"]), np.sum(info_dict["nll"])))
                else:
                    print("#{:4d} | ELBO Loss: {:7.2f} | Accuracy: {:6.2f} % | KL: {:7.2f} | NLL: {:7.2f} |"\
                          .format(epoch+1, np.sum(info_dict["nll"]) + np.sum(info_dict["kl"]), info_dict["accuracy"], \
                                  np.sum(info_dict["kl"]), np.sum(info_dict["nll"])))

        end = time.time() - start
        print("\nTraining time: {} h {} min {} s".format(int(end/3600), int((end/60)%60), int(end%60)))

        self.train_data = {
            'accuracy_train': accuracy_train,
            'accuracy_test': accuracy_test,
            'nll': nll,
            'kl': kl,
            'state_dict': self.net.module.state_dict() if len(self.device_ids) > 1 else self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'time': end,
        }

    def train_epoch(self, criterion, optimizer, beta, epoch, warmup_epochs=0, n=1):
        self.net.train()

        if epoch < warmup_epochs:
            beta /= warmup_epochs - epoch

        correct = 0
        kl_l = []
        nll_l = []
        beta = beta.to(self.device)

        for i, (input, target) in enumerate(self.train_loader):
            input = input.to(self.device)
            target = target.to(self.device)

            # compute output
            if n == 1:
                output = self.net(input)
            else:
                output = torch.zeros((input.shape[0], self.net.n_outputs)).to(self.device)
                for _ in range(n):
                    output += self.net(input)
                output /= n

            nll = criterion(output, target)
            kl = self.net.module.kl.to(self.device) if len(self.device_ids) > 1 else self.net.kl.to(self.device)
            kl *= beta
            ELBOloss = nll + kl

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            ELBOloss.backward()
            optimizer.step()

            # count correct predictions
            _, labels = output.max(1)
            correct += labels.eq(target).sum().float().item()

            # keep track of training measures
            kl_l.append(kl.item())
            nll_l.append(nll.item())

        return {"nll": nll_l,
                "kl": kl_l,
                "accuracy": correct * 100 / len(self.train_loader.dataset),}

    def evaluate(self, data_loader):
        self.net.eval()

        correct = 0
        with torch.no_grad():
            for input, target in data_loader:
                input = input.to(self.device)
                target = target.to(self.device)

                # compute output
                output = self.net(input)
                _, labels = output.max(1)

                # count correct predictions
                correct += labels.eq(target).sum()

        return correct.float() * 100 / len(data_loader.dataset)

    def save(self, path):
        self.train_data['batch_size'] = self.train_loader.batch_size
        torch.save(self.train_data, path)

class Predictor(object):
    def __init__(self,
                net,
                n_classes,
                device_ids=None):
        super(Predictor, self).__init__()

        self.device_ids = device_ids
        use_cuda = torch.cuda.is_available() and device_ids is not None
        self.device = torch.device("cuda:{}".format(device_ids[0]) if use_cuda else "cpu")
        self.net = net.to(self.device)
        self.n_classes = n_classes

    def predict(self, data_loader, n_samples, activation=torch.nn.Softmax(dim=1), variance=None):
        self.net.eval()

        batch_size = data_loader.batch_size
        len_dataset = len(data_loader.dataset)

        total = n_samples * len_dataset
        current = 0

        start = time.time()

        sys.stdout.write("[" + " "*50 + "] {:6.2f}% | {} min {} s".format(0, 0, 0))

        samples = np.empty((n_samples, len_dataset, self.n_classes))

        with torch.no_grad():
            for i, (input, _) in enumerate(data_loader):
                input = input.to(self.device)
                for n in range(n_samples):
                    samples[n, i*batch_size:min(len_dataset, (i+1)*batch_size), :] = activation(self.net(input)).data.cpu().numpy()

                    current += len(input)
                    perc = float(current)/total*100

                    sys.stdout.write("\r[" + "-" * int(perc/2) + " " * (50-int(perc/2)) + "] {:6.2f}% | {} min {} s"\
                                     .format(perc, int((time.time()-start)/60), int((time.time()-start)%60)))

        outputs = np.mean(samples, axis=0)
        variances = np.array([self.output_variance(samples[:,i,:], output, variance) for i, output in enumerate(outputs)])
        labels = np.argmax(outputs, axis=1)

        return labels, variances, outputs, samples

    @staticmethod
    def output_variance(p, p_mean, variance):
        aleatoric = np.mean(p - np.square(p), axis=0)
        epistemic = np.mean(np.square(p - np.tile(p_mean, (len(p), 1))), axis=0)
        if variance == 'top':
            aleatoric = aleatoric[np.argmax(p_mean)]
            epistemic = epistemic[np.argmax(p_mean)]
        elif variance == 'sum':
            aleatoric = np.sum(aleatoric, axis=1)
            epistemic = np.sum(epistemic, axis=1)
        return aleatoric + epistemic, aleatoric, epistemic

    @staticmethod
    def accuracy(labels, targets):
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        return np.sum((labels == targets))*100.0/len(labels)
