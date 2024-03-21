from matplotlib import pyplot as plt
from pymoo.core.callback import Callback


class moo_callback(Callback):

    def __init__(self, gb_f1, gb_f2, plot_path) -> None:
        super().__init__()
        self.data["opt_F"] = []
        self.data["pop_F"] = []
        self.data["n_evals"] = []

        self.gb_f1 = gb_f1
        self.gb_f2 = gb_f2
        self.plot_path = plot_path

    def notify(self, algorithm):
        self.data["opt_F"].append(algorithm.opt.get("F"))
        self.data["pop_F"].append(algorithm.pop.get("F"))
        self.data["n_evals"].append(algorithm.evaluator.n_eval)

        if algorithm.n_iter:
            plt.figure(figsize=(7, 7))
            PARETO = algorithm.problem.pareto_front()
            if PARETO is not None:
                plt.plot(PARETO[:, 0], PARETO[:, 1], label="Pareto Front line")
            plt.scatter(
                algorithm.pop.get("F")[:, 0],
                algorithm.pop.get("F")[:, 1],
                # color="black",
                label="Population",
                facecolor="none",
                edgecolor="black",
                marker="s",
                s=45,
            )
            plt.scatter(
                algorithm.opt.get("F")[:, 0],
                algorithm.opt.get("F")[:, 1],
                color="red",
                label="Pareto Front",
                s=20,
            )

            gb_precision, gb_recall = (
                self.gb_f1[0],
                self.gb_f2[0],
            )
            plt.scatter(
                gb_precision,
                gb_recall,
                color="blue",
                label="Baseline",
                s=20,
            )
            gb_precision, gb_recall = (
                self.gb_f1[1],
                self.gb_f2[1],
            )
            plt.scatter(
                gb_precision,
                gb_recall,
                color="green",
                label="Blocked Baseline",
                s=20,
            )
            # plt.title(shared_link)
            plt.grid()
            plt.ylabel("Retain Err")
            plt.xlabel("Forget Acc")
            plt.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=3,
            )

            plt.savefig(self.plot_path + "_paretofront.png")
            plt.close()