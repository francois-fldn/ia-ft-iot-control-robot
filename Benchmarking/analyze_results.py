#!/usr/bin/env python3
"""
Script d'analyse et visualisation des résultats de benchmarking
Génère des graphiques et un rapport HTML
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np


class BenchmarkAnalyzer:
    """Analyseur de résultats de benchmarking"""
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: Chemin vers le fichier JSON de résultats
        """
        self.results_path = Path(results_path)
        self.df = self._load_results()
        self.output_dir = self.results_path.parent / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration du style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def _load_results(self) -> pd.DataFrame:
        """Charge les résultats"""
        if self.results_path.suffix == '.json':
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif self.results_path.suffix == '.csv':
            return pd.read_csv(self.results_path)
        else:
            raise ValueError(f"Format non supporté: {self.results_path.suffix}")
    
    def plot_inference_time_comparison(self):
        """Compare les temps d'inférence entre modèles"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Préparer les données
        df_sorted = self.df.sort_values('inference_time_mean')
        
        # Créer des labels courts
        df_sorted['label'] = df_sorted.apply(
            lambda x: f"{x['model_name'].replace('.tflite', '')}\n({x['input_size']})", 
            axis=1
        )
        
        # Barplot avec barres d'erreur
        x_pos = np.arange(len(df_sorted))
        bars = ax.bar(x_pos, df_sorted['inference_time_mean'], 
                     yerr=df_sorted['inference_time_std'],
                     capsize=5, alpha=0.7)
        
        # Colorer selon le type
        colors = []
        for _, row in df_sorted.iterrows():
            if row['is_edgetpu']:
                colors.append('green')
            elif 'pruned' in row['model_type']:
                colors.append('orange')
            else:
                colors.append('blue')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Temps d\'inférence (ms)', fontsize=12)
        ax.set_title('Comparaison des temps d\'inférence', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=9)
        
        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Standard'),
            Patch(facecolor='orange', label='Pruned'),
            Patch(facecolor='green', label='EdgeTPU')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        save_path = self.output_dir / "inference_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_fps_comparison(self):
        """Compare les FPS entre modèles"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_sorted = self.df.sort_values('fps_mean', ascending=False)
        
        df_sorted['label'] = df_sorted.apply(
            lambda x: f"{x['model_name'].replace('.tflite', '')}\n({x['input_size']})", 
            axis=1
        )
        
        x_pos = np.arange(len(df_sorted))
        bars = ax.bar(x_pos, df_sorted['fps_mean'], alpha=0.7)
        
        # Colorer
        colors = []
        for _, row in df_sorted.iterrows():
            if row['is_edgetpu']:
                colors.append('green')
            elif 'pruned' in row['model_type']:
                colors.append('orange')
            else:
                colors.append('blue')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('FPS', fontsize=12)
        ax.set_title('Comparaison des FPS', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=9)
        
        # Ligne de référence 30 FPS
        ax.axhline(y=30, color='red', linestyle='--', label='30 FPS (temps réel)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        save_path = self.output_dir / "fps_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_memory_usage(self):
        """Compare l'utilisation mémoire"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_sorted = self.df.sort_values('memory_usage_mean')
        
        df_sorted['label'] = df_sorted.apply(
            lambda x: f"{x['model_name'].replace('.tflite', '')}", 
            axis=1
        )
        
        x_pos = np.arange(len(df_sorted))
        ax.bar(x_pos, df_sorted['memory_usage_mean'], alpha=0.7, label='Moyenne')
        ax.scatter(x_pos, df_sorted['memory_usage_max'], color='red', marker='x', s=100, label='Max')
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Utilisation mémoire (MB)', fontsize=12)
        ax.set_title('Utilisation mémoire', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / "memory_usage.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_cpu_usage(self):
        """Compare l'utilisation CPU"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_sorted = self.df.sort_values('cpu_usage_mean')
        
        df_sorted['label'] = df_sorted.apply(
            lambda x: f"{x['model_name'].replace('.tflite', '')}", 
            axis=1
        )
        
        x_pos = np.arange(len(df_sorted))
        ax.bar(x_pos, df_sorted['cpu_usage_mean'], alpha=0.7, label='Moyenne')
        ax.scatter(x_pos, df_sorted['cpu_usage_max'], color='red', marker='x', s=100, label='Max')
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Utilisation CPU (%)', fontsize=12)
        ax.set_title('Utilisation CPU', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / "cpu_usage.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_temperature(self):
        """Compare les températures (si disponibles)"""
        if 'temperature_mean' not in self.df.columns:
            print("⚠ Pas de données de température disponibles")
            return
        
        df_temp = self.df.dropna(subset=['temperature_mean'])
        if df_temp.empty:
            print("⚠ Pas de données de température disponibles")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_sorted = df_temp.sort_values('temperature_mean')
        
        df_sorted['label'] = df_sorted.apply(
            lambda x: f"{x['model_name'].replace('.tflite', '')}", 
            axis=1
        )
        
        x_pos = np.arange(len(df_sorted))
        ax.bar(x_pos, df_sorted['temperature_mean'], alpha=0.7, label='Moyenne')
        ax.scatter(x_pos, df_sorted['temperature_max'], color='red', marker='x', s=100, label='Max')
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Température (°C)', fontsize=12)
        ax.set_title('Température du système', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / "temperature.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_efficiency_scatter(self):
        """Graphique scatter: FPS vs Utilisation mémoire"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        scatter = ax.scatter(self.df['memory_usage_mean'], 
                           self.df['fps_mean'],
                           s=200, alpha=0.6, c=self.df['inference_time_mean'],
                           cmap='viridis')
        
        # Annoter chaque point
        for _, row in self.df.iterrows():
            label = row['model_name'].replace('.tflite', '').replace('best-', '')
            ax.annotate(label, 
                       (row['memory_usage_mean'], row['fps_mean']),
                       fontsize=8, ha='center')
        
        ax.set_xlabel('Utilisation mémoire moyenne (MB)', fontsize=12)
        ax.set_ylabel('FPS moyen', fontsize=12)
        ax.set_title('Efficacité: FPS vs Mémoire', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temps d\'inférence (ms)', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / "efficiency_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def generate_html_report(self):
        """Génère un rapport HTML"""
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Benchmarking YOLO</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #4CAF50;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .best {{
            background-color: #c8e6c9;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>📊 Rapport de Benchmarking YOLO</h1>
    
    <div class="summary">
        <h2>Résumé</h2>
        <div class="metric">
            <span class="metric-label">Nombre de modèles testés:</span>
            <span class="metric-value">{len(self.df)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Plateforme:</span>
            <span class="metric-value">{self.df['platform'].iloc[0] if len(self.df) > 0 else 'N/A'}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Date:</span>
            <span class="metric-value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>
    
    <h2>🏆 Meilleurs modèles par catégorie</h2>
    <div class="summary">
        <p><strong>Plus rapide (FPS):</strong> {self.df.loc[self.df['fps_mean'].idxmax(), 'model_name']} 
           ({self.df['fps_mean'].max():.2f} FPS)</p>
        <p><strong>Plus faible latence:</strong> {self.df.loc[self.df['inference_time_mean'].idxmin(), 'model_name']} 
           ({self.df['inference_time_mean'].min():.2f} ms)</p>
        <p><strong>Plus économe en mémoire:</strong> {self.df.loc[self.df['memory_usage_mean'].idxmin(), 'model_name']} 
           ({self.df['memory_usage_mean'].min():.1f} MB)</p>
    </div>
    
    <h2>📈 Graphiques de comparaison</h2>
"""
        
        # Ajouter les graphiques
        plots = [
            ("inference_time_comparison.png", "Temps d'inférence"),
            ("fps_comparison.png", "FPS"),
            ("memory_usage.png", "Utilisation mémoire"),
            ("cpu_usage.png", "Utilisation CPU"),
            ("efficiency_scatter.png", "Efficacité"),
        ]
        
        for plot_file, plot_title in plots:
            plot_path = self.output_dir / plot_file
            if plot_path.exists():
                html_content += f"""
    <div class="plot">
        <h3>{plot_title}</h3>
        <img src="plots/{plot_file}" alt="{plot_title}">
    </div>
"""
        
        # Tableau détaillé
        html_content += """
    <h2>📋 Résultats détaillés</h2>
    <table>
        <thead>
            <tr>
                <th>Modèle</th>
                <th>Taille</th>
                <th>Type</th>
                <th>Inférence (ms)</th>
                <th>FPS</th>
                <th>Mémoire (MB)</th>
                <th>CPU (%)</th>
                <th>Confiance</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # Identifier les meilleurs
        best_fps_idx = self.df['fps_mean'].idxmax()
        best_inference_idx = self.df['inference_time_mean'].idxmin()
        best_memory_idx = self.df['memory_usage_mean'].idxmin()
        
        for idx, row in self.df.iterrows():
            row_class = ""
            if idx in [best_fps_idx, best_inference_idx, best_memory_idx]:
                row_class = ' class="best"'
            
            html_content += f"""
            <tr{row_class}>
                <td>{row['model_name']}</td>
                <td>{row['input_size']}</td>
                <td>{row['model_type']}{' + EdgeTPU' if row['is_edgetpu'] else ''}</td>
                <td>{row['inference_time_mean']:.2f} ± {row['inference_time_std']:.2f}</td>
                <td>{row['fps_mean']:.2f}</td>
                <td>{row['memory_usage_mean']:.1f}</td>
                <td>{row['cpu_usage_mean']:.1f}</td>
                <td>{row['confidence_mean']:.3f}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
    

</body>
</html>
"""
        
        # Sauvegarder
        report_path = self.results_path.parent / f"benchmark_report_{self.df['timestamp'].iloc[0]}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Rapport HTML généré: {report_path}")
    
    def analyze(self):
        """Lance toutes les analyses"""
        print(f"\n{'='*60}")
        print(f"Analyse des résultats: {self.results_path.name}")
        print(f"{'='*60}\n")
        
        print("Génération des graphiques...")
        self.plot_inference_time_comparison()
        self.plot_fps_comparison()
        self.plot_memory_usage()
        self.plot_cpu_usage()
        self.plot_temperature()
        self.plot_efficiency_scatter()
        
        print("\nGénération du rapport HTML...")
        self.generate_html_report()
        
        print(f"\n{'='*60}")
        print(f"Analyse terminée!")
        print(f"Graphiques dans: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyse des résultats de benchmarking")
    parser.add_argument('results', type=str, 
                       help='Chemin vers le fichier de résultats (JSON ou CSV)')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results)
    analyzer.analyze()


if __name__ == '__main__':
    main()
