"""
Feature Extractor - batch processing of images.

Provides easy extraction of multiple features from single images or folders.
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Type
import numpy as np

from imagefeatures.base import GlobalFeature, list_features
from imagefeatures.utils import load_image, get_image_files


class FeatureExtractor:
    """
    Extract multiple features from images.
    
    Supports single images, lists of images, and entire folders.
    Output can be saved as CSV or pickle (sklearn-compatible).
    
    Example:
        >>> from imagefeatures import FeatureExtractor
        >>> from imagefeatures.features import ColorHistogram, LBP, Gabor
        >>> 
        >>> # Create extractor with specific features
        >>> extractor = FeatureExtractor([ColorHistogram(), LBP(), Gabor()])
        >>> 
        >>> # Extract from single image
        >>> features = extractor.extract("image.jpg")
        >>> 
        >>> # Extract from folder and save as CSV
        >>> extractor.extract_folder("./images/", output="features.csv")
        >>> 
        >>> # Extract and save as sklearn-compatible pickle
        >>> extractor.extract_folder("./images/", output="features.pkl")
    """
    
    def __init__(self, features: Optional[List[GlobalFeature]] = None):
        """
        Initialize extractor with list of feature instances.
        
        Args:
            features: List of GlobalFeature instances. If None, uses all registered features.
        """
        if features is None:
            # Import to trigger registration
            import imagefeatures.features
            # Create instances of all registered features
            self.features = [cls() for cls in list_features().values()]
        else:
            self.features = features
    
    def extract(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Extract all features from a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict mapping feature names to feature vectors
        """
        image = load_image(image_path)
        results = {}
        
        for feature in self.features:
            feature.extract(image)
            results[feature.name] = feature.get_feature_vector().copy()
        
        return results
    
    def extract_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, np.ndarray]]:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dicts, each mapping feature names to vectors
        """
        return [self.extract(path) for path in image_paths]
    
    def extract_folder(
        self, 
        folder_path: Union[str, Path],
        output: Optional[Union[str, Path]] = None,
        combine_features: bool = True
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract features from all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output: Optional output file path (.csv or .pkl)
            combine_features: If True, concatenate all features into single vector
            
        Returns:
            Dict with 'features' (2D array) and 'filenames' (list of names)
        """
        folder = Path(folder_path)
        image_files = get_image_files(folder)
        
        if not image_files:
            raise ValueError(f"No image files found in {folder}")
        
        all_features = []
        filenames = []
        
        for img_path in image_files:
            try:
                result = self.extract(img_path)
                
                if combine_features:
                    # Concatenate all features into single vector
                    combined = np.concatenate(list(result.values()))
                    all_features.append(combined)
                else:
                    all_features.append(result)
                
                filenames.append(img_path.name)
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
        
        if combine_features:
            feature_matrix = np.array(all_features)
        else:
            feature_matrix = all_features
        
        result = {
            'features': feature_matrix,
            'filenames': filenames,
            'feature_names': [f.name for f in self.features],
        }
        
        # Save output if requested
        if output:
            output = Path(output)
            if output.suffix.lower() == '.csv':
                self._save_csv(result, output, combine_features)
            elif output.suffix.lower() in ('.pkl', '.pickle'):
                self._save_pickle(result, output)
            else:
                raise ValueError(f"Unknown output format: {output.suffix}")
        
        return result
    
    def _save_csv(self, result: dict, output_path: Path, combined: bool) -> None:
        """Save features to CSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            if combined:
                # Generate column names from feature dimensions
                header = ['filename']
                for feature in self.features:
                    dim = feature.dim
                    for i in range(dim):
                        header.append(f"{feature.name}_{i}")
                writer.writerow(header)
                
                # Data rows
                for i, filename in enumerate(result['filenames']):
                    row = [filename] + list(result['features'][i])
                    writer.writerow(row)
            else:
                # One column per feature (flatten each)
                header = ['filename'] + result['feature_names']
                writer.writerow(header)
                
                for i, filename in enumerate(result['filenames']):
                    row = [filename]
                    for feat_dict in result['features'][i].values():
                        row.extend(feat_dict.flatten())
                    writer.writerow(row)
    
    def _save_pickle(self, result: dict, output_path: Path) -> None:
        """
        Save features as pickle file.
        
        Format is sklearn-compatible: (X, filenames) tuple where X is 2D array.
        """
        with open(output_path, 'wb') as f:
            # Save as tuple (feature_matrix, filenames) for sklearn compatibility
            pickle.dump((result['features'], result['filenames']), f)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]
    
    def get_total_dimensions(self) -> int:
        """Get total number of feature dimensions when combined."""
        return sum(f.dim for f in self.features)
