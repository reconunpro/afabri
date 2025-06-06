import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from pathlib import Path

class WallAnalyzer:
    def _filter_data(self, data, min_thickness=0.01, max_thickness=3.0):
        """Filter out unrealistic thickness values"""
        data_array = np.asarray(data)
        return data_array[(data_array >= min_thickness) & (data_array <= max_thickness)]

    def _find_wall_thickness_by_clustering(self, data, n_clusters=5):
        """Find wall thickness using K-means clustering"""
        
        # Reshape for K-means
        X = data.reshape(-1, 1)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        # Get cluster centers and sizes
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort by center value
        sorted_indices = np.argsort(centers)
        sorted_centers = centers[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # Calculate percentage of each cluster
        total = sum(sorted_counts)
        percentages = sorted_counts / total * 100
        
        results = []
        for i, (center, count, pct) in enumerate(zip(sorted_centers, sorted_counts, percentages)):
            results.append({
                'cluster': i + 1,
                'thickness': round(center, 3),
                'count': int(count),
                'percentage': round(pct, 2)
            })
        
        return results, sorted_indices, sorted_centers, labels

    def analyze(self, ray_data, plot=True, print_results=True, find_significant=True):
        """
        Analyze wall thickness using ray-casting data
        
        Args:
            ray_data: Array of ray-casting measurements
            plot: Whether to create visualizations
            print_results: Whether to print analysis results
            find_significant: If True, find thinnest SIGNIFICANT wall (default behavior).
                              If False, find the absolute thinnest measurement.
        """
        
        # Filter data first
        filter_min = 0.01
        filter_max = 5.0
        ray_data = self._filter_data(ray_data, filter_min, filter_max)
        if len(ray_data) == 0:
            print("No valid data after filtering.")
            return None
        
        # Perform clustering analysis
        ray_clusters, sorted_indices, sorted_centers, labels = self._find_wall_thickness_by_clustering(ray_data)
        if print_results:
            print("\nK-Means Clustering Analysis")
            print("\nRay-casting Clusters:")
            print(pd.DataFrame(ray_clusters))
        
        # Determine thinnest wall based on mode
        thinnest_wall = None
        
        if find_significant:
            # Find thinnest significant wall (skip noise clusters)
            for i, cluster in enumerate(ray_clusters):
                if cluster['percentage'] > 5:  # Skip likely noise
                    cluster_index = sorted_indices[i]
                    cluster_points = ray_data[labels == cluster_index]
                    
                    if len(cluster_points) > 0:
                        thinnest_wall = self._find_thinnest_significant_wall(cluster_points, plot=False)  # Disable subplot plotting
                        if print_results:
                            print(f"Cluster center: {cluster['thickness']:.3f}, thinnest significant: {thinnest_wall:.3f}")
                    break
        else:
            # Find absolute thinnest measurement
            thinnest_wall = np.min(ray_data)
            if print_results:
                print(f"Absolute thinnest measurement: {thinnest_wall:.3f}")

        # Create visualization if requested
        if plot:
            try:
                self._plot_wall_thickness_analysis(
                    ray_data, ray_clusters, sorted_indices, sorted_centers, labels, thinnest_wall
                )
            except Exception as e:
                print(f"Warning: Plotting failed: {e}")

        return thinnest_wall

    def _plot_wall_thickness_analysis(self, ray_data, ray_clusters, sorted_indices, sorted_centers, labels, thinnest_wall):
        """Create visualization of wall thickness analysis"""
        
        if len(ray_data) == 0:
            print("Warning: Not enough valid ray data points for visualization")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Create a unique color for each cluster
            n_clusters = len(sorted_centers)
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
            
            # Plot each cluster
            for i, (center, color) in enumerate(zip(sorted_centers, colors)):
                cluster_data = ray_data[labels == sorted_indices[i]]
                plt.hist(cluster_data, bins=20, alpha=0.6, color=color, 
                        label=f'Cluster {i+1}: {center:.3f} ({ray_clusters[i]["percentage"]:.1f}%)')
                plt.axvline(x=center, color=color, linestyle='--', linewidth=2)
            
            # Highlight the thinnest wall
            if thinnest_wall is not None:
                plt.axvline(x=thinnest_wall, color='red', linestyle='-', linewidth=3, 
                        label=f'Thinnest wall: {thinnest_wall:.3f}')
            
            plt.title('Ray-casting Wall Thickness Analysis')
            plt.xlabel('Wall Thickness')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create safe directory and save
            output_dir = Path('visualization')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / 'wall_thickness_analysis.png'
            
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wall thickness plot saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating wall thickness plot: {e}")
            plt.close()  # Make sure to close figure even if save fails

    def _find_thinnest_significant_wall(self, cluster_points, min_significance_pct=15, plot=False):
        """
        Find the thinnest wall thickness that still has significant frequency in the cluster.
        
        Args:
            cluster_points: Array of thickness values belonging to a cluster
            min_significance_pct: Minimum percentage of points needed to consider a bin significant
            plot: If True, creates visualization of the histogram and threshold
            
        Returns:
            The thinnest wall thickness with significant frequency
        """
        
        if len(cluster_points) < 5:
            return np.min(cluster_points)
        
        # Create histogram with adaptive bin size
        data_range = np.max(cluster_points) - np.min(cluster_points)
        n_bins = min(30, max(10, int(len(cluster_points) / 5)))
        
        hist, bin_edges = np.histogram(cluster_points, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate significance threshold
        total_points = len(cluster_points)
        min_count = max(5, int(total_points * min_significance_pct / 100.0))
        
        # Find the lowest bin with significant frequency
        selected_threshold = None
        for i in range(len(hist)):
            if hist[i] >= min_count:
                selected_threshold = bin_centers[i]
                break
        
        # Fallback to cluster center if no significant bin found
        if selected_threshold is None:
            max_index = np.argmax(hist)
            selected_threshold = bin_centers[max_index]
        
        # Create visualization if requested
        if plot:
            try:
                plt.figure(figsize=(10, 6))
                
                # Plot histogram
                plt.hist(cluster_points, bins=bin_edges, alpha=0.7, color='skyblue')
                
                # Mark significance threshold
                plt.axvline(x=selected_threshold, color='r', linestyle='--', linewidth=2,
                            label=f'Threshold: {selected_threshold:.3f}')
                
                # Add horizontal line for minimum count
                plt.axhline(y=min_count, color='green', linestyle=':', linewidth=2,
                            label=f'Min Count: {min_count} ({min_significance_pct}%)')
                
                plt.title('Wall Thickness Distribution in Selected Cluster')
                plt.xlabel('Thickness')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Annotate bins with their counts
                for i, count in enumerate(hist):
                    if count > 0:
                        plt.text(bin_centers[i], count + 0.5, str(count), 
                                ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                
                # Create safe directory and save
                output_dir = Path('visualization')
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / 'wall_cluster_selection.png'
                
                plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Cluster selection plot saved to: {output_path}")
                
            except Exception as e:
                print(f"Error creating cluster selection plot: {e}")
                plt.close()  # Make sure to close figure even if save fails
        
        return selected_threshold