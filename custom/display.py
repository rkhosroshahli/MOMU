import numpy as np
from pymoo.util.display.column import Column
from pymoo.util.display.multi import MultiObjectiveOutput


class moo_display(MultiObjectiveOutput):
    def __init__(self, evaluator, unblocker, test_loader):
        super().__init__()
        self.ave_precision = Column("average forget acc (f1)", width=25)
        self.ave_recall = Column("average retain acc (f2)", width=25)
        self.best_precision = Column("best forget acc (f1)", width=20)
        self.best_recall = Column("best retain acc (f2)", width=20)
        # self.train_best_f1 = Column("best train f1-score", width=20)
        # self.train_ave_f1 = Column("ave train f1-score", width=20)
        self.val_best_f1 = Column("best test f1-score", width=20)
        self.val_ave_f1 = Column("ave test f1-score", width=20)

        self.evaluator = evaluator
        self.test_loader = test_loader
        self.unblocker = unblocker

    def initialize(self, algorithm):
        super().initialize(algorithm)
        self.columns += [
            self.best_precision,
            self.ave_precision,
            self.best_recall,
            self.ave_recall,
            # self.train_best_f1,
            # self.train_ave_f1,
            self.val_best_f1,
            self.val_ave_f1,
        ]

    def update(self, algorithm):
        super().update(algorithm)
        self.ave_precision.set(
            f"{np.mean(algorithm.opt.get('F')[:, 0]):.6f} (\u00B1{np.std(algorithm.opt.get('F')[:, 0]):.6f})"
        )
        self.ave_recall.set(
            f"{np.mean(1 - algorithm.opt.get('F')[:, 1]):.6f} (\u00B1{np.std(1 - algorithm.opt.get('F')[:, 1]):.6f})"
        )
        self.best_precision.set(
            f"{np.min(algorithm.opt.get('F')[:, 0]):.6f}"
        )
        self.best_recall.set(
            f"{np.max(1 - algorithm.opt.get('F')[:, 1]):.6f}"
        )

        if algorithm.n_iter % 5 == 0 or algorithm.n_iter == 1:
            pf_val, pf_train = [], []
            pf = algorithm.opt.get("X")

            # print(pf[:5])
            for i in range(pf.shape[0]):
                pfi = pf[i]
                if self.unblocker is not None:
                    pfi, = self.unblocker(np.array([pf[i]]))
                val = self.evaluator(pfi, self.test_loader)
                # print(val)
                pf_val.append(val)

            # self.train_best_f1.set(f"{np.max(pf_train):.6f}")
            # self.train_ave_f1.set(f"{np.mean(pf_train):.6f}")
            self.val_best_f1.set(f"{np.max(pf_val):.6f}")
            self.val_ave_f1.set(f"{np.mean(pf_val):.6f}")

        else:
            # self.train_best_f1.set(f"-")
            # self.train_ave_f1.set(f"-")
            self.val_best_f1.set(f"-")
            self.val_ave_f1.set(f"-")
