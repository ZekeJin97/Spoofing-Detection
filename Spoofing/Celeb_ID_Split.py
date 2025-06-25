import os
import re
import random
import shutil
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob


class BalancedDatasetCreator:
    """
    Creates a balanced dataset ensuring no ID leakage between train/test splits
    """

    def __init__(self, real_dir, fake_dir, output_dir="balanced_dataset",
                 max_samples_per_class=5000, frames_per_id=10, test_size=0.2):
        """
        Args:
            real_dir: Directory containing real images
            fake_dir: Directory containing fake images
            output_dir: Output directory for balanced dataset
            max_samples_per_class: Maximum samples per class (real/fake)
            frames_per_id: Number of frames to keep per ID (reduces redundancy)
            test_size: Proportion for test set
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.output_dir = output_dir
        self.max_samples_per_class = max_samples_per_class
        self.frames_per_id = frames_per_id
        self.test_size = test_size

    def extract_id_from_filename(self, filename):
        """Extract the TARGET person ID (first ID in filename)"""
        # For real files: real_id0_0000_frame0000.jpg -> target person ID is "0"
        real_match = re.search(r'real_id(\d+)_', filename)
        if real_match:
            return real_match.group(1)  # Return the target person ID

        # For fake files: fake_id0_id16_0000_frame0000.jpg -> target person ID is "0"
        # (id0 is the target, id16 is the source used for deepfake)
        fake_match = re.search(r'fake_id(\d+)_', filename)
        if fake_match:
            return fake_match.group(1)  # Return the TARGET person ID (first one)

        # Fallback: extract first ID found
        matches = re.findall(r'id(\d+)', filename)
        if matches:
            return matches[0]  # Always return the first one (target identity)
        return "unknown"

    def debug_id_extraction(self, sample_size=10):
        """Debug the ID extraction to verify it's working correctly"""
        print("🔍 Debugging ID extraction...")

        real_files = glob(os.path.join(self.real_dir, "*"))[:sample_size]
        fake_files = glob(os.path.join(self.fake_dir, "*"))[:sample_size]

        print("\n📋 Real file ID extraction:")
        for file_path in real_files:
            filename = os.path.basename(file_path)
            extracted_id = self.extract_id_from_filename(filename)
            print(f"   {filename} -> ID: {extracted_id}")

        print("\n📋 Fake file ID extraction:")
        for file_path in fake_files:
            filename = os.path.basename(file_path)
            extracted_id = self.extract_id_from_filename(filename)
            print(f"   {filename} -> ID: {extracted_id}")

    def analyze_dataset(self):
        """Analyze the current dataset structure"""
        print("🔍 Analyzing dataset structure...")

        # Get all image files
        real_files = glob(os.path.join(self.real_dir, "*"))
        fake_files = glob(os.path.join(self.fake_dir, "*"))

        print(f"📊 Found {len(real_files)} real files")
        print(f"📊 Found {len(fake_files)} fake files")

        # Group by ID
        real_ids = defaultdict(list)
        fake_ids = defaultdict(list)

        for file_path in real_files:
            filename = os.path.basename(file_path)
            id_key = self.extract_id_from_filename(filename)
            real_ids[id_key].append(file_path)

        for file_path in fake_files:
            filename = os.path.basename(file_path)
            id_key = self.extract_id_from_filename(filename)
            fake_ids[id_key].append(file_path)

        print(f"📊 Real IDs: {len(real_ids)} unique IDs")
        print(f"📊 Fake IDs: {len(fake_ids)} unique IDs")

        # Check for overlapping IDs (potential leakage)
        real_id_set = set(real_ids.keys())
        fake_id_set = set(fake_ids.keys())
        overlap = real_id_set.intersection(fake_id_set)

        if overlap:
            print(f"🚨 CRITICAL: {len(overlap)} overlapping person IDs found!")
            print(f"   Overlapping IDs: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
            print(f"   ⚠️  This is IDENTITY LEAKAGE - same people in both real and fake!")
            print(f"   📋 Strategy: Each person will be assigned to ONLY ONE class")
        else:
            print("✅ No overlapping IDs found")

        # Show distribution
        real_counts = [len(files) for files in real_ids.values()]
        fake_counts = [len(files) for files in fake_ids.values()]

        if real_counts:
            print(
                f"📈 Real frames per ID: min={min(real_counts)}, max={max(real_counts)}, avg={np.mean(real_counts):.1f}")
        else:
            print("📈 Real frames per ID: No real files found")

        if fake_counts:
            print(
                f"📈 Fake frames per ID: min={min(fake_counts)}, max={max(fake_counts)}, avg={np.mean(fake_counts):.1f}")
        else:
            print("📈 Fake frames per ID: No fake files found")

        return real_ids, fake_ids, overlap

    def select_balanced_samples(self, id_groups, label):
        """Select balanced samples from ID groups"""
        selected_files = []

        # Convert to list of (id, files) and shuffle
        id_list = list(id_groups.items())
        random.shuffle(id_list)

        for id_key, files in id_list:
            if len(selected_files) >= self.max_samples_per_class:
                break

            # Sort files to get consistent frame selection
            files.sort()

            # Select up to frames_per_id frames per ID
            selected_from_id = files[:self.frames_per_id]
            selected_files.extend(selected_from_id)

            if len(selected_files) >= self.max_samples_per_class:
                selected_files = selected_files[:self.max_samples_per_class]
                break

        print(
            f"✅ Selected {len(selected_files)} {label} samples from {len(set(self.extract_id_from_filename(os.path.basename(f)) for f in selected_files))} unique IDs")
        return selected_files

    def create_balanced_dataset(self):
        """Create the balanced dataset with person-level train/test split for deepfakes"""
        print("🚀 Creating balanced dataset...")

        # First debug the ID extraction
        self.debug_id_extraction()

        # Analyze current dataset
        real_ids, fake_ids, overlap = self.analyze_dataset()

        # For deepfake datasets, we expect 100% overlap
        if len(overlap) == len(real_ids) or len(overlap) > len(real_ids) * 0.8:
            print("🎯 DETECTED: Deepfake dataset (same people in real and fake)")
            split_result = self.handle_deepfake_dataset_split(real_ids, fake_ids, overlap)

            # Create output directories
            os.makedirs(os.path.join(self.output_dir, "train", "real"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "train", "fake_generated"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "test", "real"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "test", "fake_generated"), exist_ok=True)

            # Copy files
            print("📁 Copying files...")
            self.copy_files(split_result['train_real'], os.path.join(self.output_dir, "train", "real"))
            self.copy_files(split_result['train_fake'], os.path.join(self.output_dir, "train", "fake_generated"))
            self.copy_files(split_result['test_real'], os.path.join(self.output_dir, "test", "real"))
            self.copy_files(split_result['test_fake'], os.path.join(self.output_dir, "test", "fake_generated"))

            # Create summary
            summary = {
                'total_samples': len(split_result['train_real']) + len(split_result['train_fake']) + len(
                    split_result['test_real']) + len(split_result['test_fake']),
                'train_samples': len(split_result['train_real']) + len(split_result['train_fake']),
                'test_samples': len(split_result['test_real']) + len(split_result['test_fake']),
                'train_people': len(split_result['train_people']),
                'test_people': len(split_result['test_people']),
                'frames_per_id': self.frames_per_id,
                'dataset_type': 'deepfake_person_split'
            }

            return summary

        else:
            print("🎯 DETECTED: Standard dataset (different people in real vs fake)")
            # Use the original logic for standard datasets
            return self.handle_standard_dataset_split(real_ids, fake_ids, overlap)

    def handle_deepfake_dataset_split(self, real_ids, fake_ids, overlap):
        """Handle person-level splitting for deepfake datasets"""

    def handle_deepfake_dataset_split(self, real_ids, fake_ids, overlap):
        """Handle person-level splitting for deepfake datasets"""
        print(f"🔧 Resolving identity leakage for {len(overlap)} overlapping persons...")
        print("🎯 DEEPFAKE DATASET STRATEGY: Person-level train/test split")
        print("   Same person appears in both real and fake classes")
        print("   Solution: Each person goes ENTIRELY to either train OR test")

        # Convert overlapping people to a list and shuffle
        all_people = list(overlap)
        random.shuffle(all_people)

        # Calculate train/test split for people
        n_test_people = max(1, int(len(all_people) * self.test_size))
        n_train_people = len(all_people) - n_test_people

        train_people = set(all_people[:n_train_people])
        test_people = set(all_people[n_train_people:])

        print(f"   📊 {len(train_people)} people assigned to TRAIN")
        print(f"   📊 {len(test_people)} people assigned to TEST")

        # Separate files based on person assignment
        train_real_files = []
        train_fake_files = []
        test_real_files = []
        test_fake_files = []

        # Process real files
        for person_id, files in real_ids.items():
            random.shuffle(files)  # 🔀 Add this
            if person_id in train_people:
                train_real_files.extend(files[:self.frames_per_id])
            elif person_id in test_people:
                test_real_files.extend(files[:self.frames_per_id])

        # Process fake files
        for person_id, files in fake_ids.items():
            random.shuffle(files)  # 🔀 Add this
            if person_id in train_people:
                train_fake_files.extend(files[:self.frames_per_id])
            elif person_id in test_people:
                test_fake_files.extend(files[:self.frames_per_id])

        # Balance classes within train and test
        min_train = min(len(train_real_files), len(train_fake_files))
        min_test = min(len(test_real_files), len(test_fake_files))

        # Limit to max_samples_per_class
        max_train_per_class = min(min_train, self.max_samples_per_class)
        max_test_per_class = min(min_test, self.max_samples_per_class // 4)  # Smaller test set

        train_real_files = train_real_files[:max_train_per_class]
        train_fake_files = train_fake_files[:max_train_per_class]
        test_real_files = test_real_files[:max_test_per_class]
        test_fake_files = test_fake_files[:max_test_per_class]

        print(f"   ✅ Final dataset:")
        print(f"   📊 Train: {len(train_real_files)} real + {len(train_fake_files)} fake")
        print(f"   📊 Test: {len(test_real_files)} real + {len(test_fake_files)} fake")
        print(f"   🎯 Zero identity leakage - no person appears in both train and test!")

        return {
            'train_real': train_real_files,
            'train_fake': train_fake_files,
            'test_real': test_real_files,
            'test_fake': test_fake_files,
            'train_people': train_people,
            'test_people': test_people
        }

    def copy_files(self, file_list, dest_dir):
        """Copy files to destination directory"""
        for src_file in file_list:
            dst_file = os.path.join(dest_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)

    def update_pipeline_paths(self, pipeline_file="improved_pipeline.py"):
        """Update the pipeline to use the new balanced dataset"""
        if not os.path.exists(pipeline_file):
            print(f"⚠️  Pipeline file {pipeline_file} not found")
            return

        # Read the pipeline file
        with open(pipeline_file, 'r') as f:
            content = f.read()

        # Update dataset path
        updated_content = content.replace(
            'load_dataset("dataset")',
            f'load_dataset("{self.output_dir}/train")'
        )

        # Save updated pipeline
        updated_file = pipeline_file.replace('.py', '_balanced.py')
        with open(updated_file, 'w') as f:
            f.write(updated_content)

        print(f"✅ Updated pipeline saved as {updated_file}")


def main():
    """Main function to create balanced dataset"""
    # Configuration - adjust these parameters as needed
    config = {
        'real_dir': 'dataset/celeb_real',
        'fake_dir': 'dataset/celeb_fake',
        'output_dir': 'balanced_dataset',
        'max_samples_per_class': 1000,  # keep it light
        'frames_per_id': 10,  # avoid spammy video frames
        'test_size': 0.2  # 80/20 split
    }

    # Create dataset creator
    creator = BalancedDatasetCreator(**config)

    # Create balanced dataset
    summary = creator.create_balanced_dataset()

    print("\n" + "=" * 50)
    print("📋 DATASET CREATION SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"\n✅ Balanced dataset created in: {config['output_dir']}")
    print("📁 Directory structure:")
    print(f"   {config['output_dir']}/")
    print("   ├── train/")
    print("   │   ├── real/")
    print("   │   └── fake_generated/")
    print("   └── test/")
    print("       ├── real/")
    print("       └── fake_generated/")

    # Update pipeline paths
    creator.update_pipeline_paths()

    return summary


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    summary = main()