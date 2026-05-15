"""Minimal example showing how to wire together nstad_bench components."""

# from nstad_bench.data.base import BaseDataLoader
# from nstad_bench.adaptation.base import BaseAdaptation
# from nstad_bench.models.base import BaseModel
# from nstad_bench.experiments.base import BaseExperiment
# from nstad_bench.metrics.base import BaseMetric

# Implement your concrete subclasses and compose them here:
#
#   loader = MyLoader()
#   dataset = loader.load(Path("data/my_dataset"))
#   source, target = MySplitter().split(dataset)
#   adapted = MyAdaptation().fit_transform(source.X, target.X)
#   model = MyModel().fit(adapted, source.y)
#   score = MyMetric()(target.y, model.predict(target.X))
#   print(score)


def main() -> None:
    raise NotImplementedError("Replace with a concrete experiment.")


if __name__ == "__main__":
    main()
