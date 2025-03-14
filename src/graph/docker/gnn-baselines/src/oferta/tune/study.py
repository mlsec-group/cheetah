import optuna


def create_study(study_name: str, db_string: str, reset: bool = False, direction: str = "minimize"):
    if reset:
        optuna.delete_study(
            study_name=study_name,
            storage=db_string,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=db_string,
        load_if_exists=True,
        direction=direction
    )
    return study
