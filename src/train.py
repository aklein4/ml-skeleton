
import hydra
import omegaconf

import datasets

import utils.constants as constants
from utils.import_utils import import_model, import_trainer


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    print("Loading model...")
    if config.model.pretrained is not None:
        
        model = import_model(config.model.type).from_pretrained(
            config.model.pretrained,
        )

    else:
        
        model.config = import_model(config.model.config.type)(
            **config.model.config.kwargs
        )
        model = import_model(config.model.type)(model.config)

    model = model.to(constants.DEVICE)
    
    print("Loading dataset...")
    dataset = datasets.load_dataset(
        config.dataset.name,
        **config.dataset.kwargs,
    )

    print("Loading trainer...")
    trainer = import_trainer(config.trainer.type)(
        config,
        model,
        dataset,
    )

    print("Entering trainer...")
    trainer.train()


if __name__ == '__main__':
    main()
    