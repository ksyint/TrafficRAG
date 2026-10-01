import argparse
import json

from .recipe import ROOT, catalog_profiles, load_recipe


def main():
    parser = argparse.ArgumentParser(description='Inspect grounding/motion recipes without loading encoders.')
    parser.add_argument('--profile')
    parser.add_argument('--validate-all', action='store_true')
    args = parser.parse_args()
    if args.profile:
        print(json.dumps(load_recipe(args.profile).to_dict(), indent=2))
    elif args.validate_all:
        paths = catalog_profiles()
        for path in paths:
            load_recipe(path)
        print(json.dumps({'validated_recipes': len(paths)}))
    else:
        print('\n'.join(path.relative_to(ROOT).as_posix() for path in catalog_profiles()))


if __name__ == '__main__':
    main()
