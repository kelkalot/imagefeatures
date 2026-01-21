"""
Command-line interface for imagefeatures.

Usage:
    imagefeatures extract <image_or_folder> [--features <list>] [-o <output>]
    imagefeatures list-features
"""

import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    try:
        import click
    except ImportError:
        print("CLI requires click. Install with: pip install click")
        print("Or install imagefeatures with: pip install imagefeatures[cli]")
        sys.exit(1)
    
    run_cli()


def run_cli():
    """Run the CLI with click."""
    import click
    
    @click.group()
    @click.version_option(version="0.1.0")
    def cli():
        """Image feature extraction tool."""
        pass
    
    @cli.command()
    def list_features():
        """List all available features."""
        from imagefeatures.base import list_features as get_features
        import imagefeatures.features  # Trigger registration
        
        features = get_features()
        click.echo("Available features:")
        for name, cls in sorted(features.items()):
            instance = cls()
            click.echo(f"  {name}: {instance.dim} dimensions")
    
    @cli.command()
    @click.argument('path', type=click.Path(exists=True))
    @click.option('--features', '-f', default=None, 
                  help='Comma-separated list of features (default: all)')
    @click.option('--output', '-o', default=None,
                  help='Output file (.csv or .pkl)')
    def extract(path, features, output):
        """Extract features from an image or folder."""
        from imagefeatures import FeatureExtractor
        from imagefeatures.base import get_feature
        import imagefeatures.features  # Trigger registration
        
        path = Path(path)
        
        # Parse feature list
        if features:
            feature_names = [f.strip() for f in features.split(',')]
            feature_instances = []
            for name in feature_names:
                cls = get_feature(name)
                if cls is None:
                    click.echo(f"Unknown feature: {name}", err=True)
                    sys.exit(1)
                feature_instances.append(cls())
        else:
            feature_instances = None  # Use all
        
        extractor = FeatureExtractor(feature_instances)
        
        if path.is_dir():
            # Folder extraction
            if output is None:
                output = path / "features.csv"
            
            click.echo(f"Extracting features from {path}...")
            result = extractor.extract_folder(path, output=output)
            click.echo(f"Processed {len(result['filenames'])} images")
            click.echo(f"Features: {', '.join(result['feature_names'])}")
            click.echo(f"Total dimensions: {result['features'].shape[1]}")
            click.echo(f"Output saved to: {output}")
        else:
            # Single image
            click.echo(f"Extracting features from {path}...")
            result = extractor.extract(path)
            
            if output:
                # Save to file
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['feature', 'values']
                    writer.writerow(header)
                    for name, vec in result.items():
                        writer.writerow([name, ','.join(map(str, vec))])
                click.echo(f"Output saved to: {output}")
            else:
                # Print to stdout
                for name, vec in result.items():
                    click.echo(f"{name}: {vec.shape[0]} dims, "
                              f"range [{vec.min():.2f}, {vec.max():.2f}]")
    
    cli()


if __name__ == '__main__':
    main()
