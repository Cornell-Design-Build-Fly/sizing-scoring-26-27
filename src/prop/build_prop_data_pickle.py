from src.prop.prop_database import (
    DEFAULT_PROP_CACHE_PATH,
    load_default_prop_database,
)


def main():
    load_default_prop_database()
    print(f"Prop interpolator cache ready at: {DEFAULT_PROP_CACHE_PATH}")


if __name__ == "__main__":
    main()