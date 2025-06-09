from src.config import Config
from src.evaluator import Evaluator
from src.evaluator.functional import Functional
from src.evaluator.likelihood import Likelihood
from src.predictor import Predictor
from src.sampler import Sampler


def main():
    cfg = Config()

    samples = None

    sampler = Sampler(cfg["sampler"])
    if sampler.sample_path.exists():
        samples = sampler.load()
    else:
        samples = sampler.sample()

    ex_max_predictor = Predictor(cfg["predictor"]["ex_max"])
    if ex_max_predictor.model_path.exists():
        ex_max_predictor.load()
    else:
        ex_max_predictor.train()

    em_max_predictor = Predictor(cfg["predictor"]["em_max"])
    if em_max_predictor.model_path.exists():
        em_max_predictor.load()
    else:
        em_max_predictor.train()

    brightness_predictor = Predictor(cfg["predictor"]["brightness"])
    if brightness_predictor.model_path.exists():
        brightness_predictor.load()
    else:
        brightness_predictor.train()

    likelihood = Likelihood(cfg["evaluator"]["likelihood"])
    ex_max = Functional(cfg["evaluator"]["ex_max"], predictor=ex_max_predictor)
    em_max = Functional(cfg["evaluator"]["em_max"], predictor=em_max_predictor)
    brightness = Functional(
        cfg["evaluator"]["brightness"], predictor=brightness_predictor
    )

    evaluator = Evaluator(likelihood, functionals=[ex_max, em_max, brightness])

    results = evaluator.filter(samples)

    print(len(results))


if __name__ == "__main__":
    main()
