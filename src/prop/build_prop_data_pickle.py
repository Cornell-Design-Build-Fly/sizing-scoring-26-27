from src.prop.prop_database import (
    DEFAULT_PROP_PICKLE_PATH,
    load_default_prop_database,
)


def main():
    load_default_prop_database()
    print(
        f"Prop interpolator cache ready at: "
        f"{DEFAULT_PROP_PICKLE_PATH}"
    )


if __name__ == "__main__":
    main()