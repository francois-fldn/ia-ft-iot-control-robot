#!/usr/bin/env python3
"""
Script de comparaison multi-plateformes
Compare les résultats de benchmarking entre différentes plateformes
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


class MultiPlatformComparator:
    """Compare les résultats entre plateformes"""
    
    def __init__(self, result_files: list):
        """
        Args:
            result_files: Liste des chemins vers les fichiers de résultats JSON
        """
        self.result_files = [Path(f) for f in result_files]
        self.df = self._load_all_results()
        self.output_dir = Path("benchmark_results/comparison")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def _load_all_results(self) -> pd.DataFrame:
        """Charge tous les résultats"""
        all_data = []
        
        for file_path in self.result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            all_data.extend(data)
        
        return pd.DataFrame(all_data)
    
    def plot_platform_comparison(self, metric: str, ylabel: str, title: str, 
                                 filename: str, lower_is_better: bool = True):
        """
        Crée un graphique de comparaison entre plateformes
        
        Args:
            metric: Nom de la métrique à comparer
            ylabel: Label de l'axe Y
            title: Titre du graphique
            filename: Nom du fichier de sortie
            lower_is_better: Si True, les valeurs plus basses sont meilleures
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Grouper par modèle et plateforme
        platforms = self.df['platform'].unique()
        
        # Prendre un sous-ensemble de modèles représentatifs
        model_types = []
        for size in [256, 320]:
            for dtype in ['int8', 'fp16', 'fp32']:
                for pruned in ['', '_pruned']:
                    pattern = f"{dtype}_{size}{pruned}"
                    model_types.append(pattern)
        
        # Filtrer les modèles qui existent
        models_to_plot = []
        for pattern in model_types:
            matching = self.df[self.df['model_name'].str.contains(pattern, case=False)]
            if not matching.empty:
                models_to_plot.append(matching.iloc[0]['model_name'])
        
        # Limiter à 10 modèles max pour la lisibilité
        models_to_plot = models_to_plot[:10]
        
        # Préparer les données
        x = np.arange(len(models_to_plot))
        width = 0.8 / len(platforms)
        
        for i, platform in enumerate(platforms):
            values = []
            for model in models_to_plot:
                row = self.df[(self.df['model_name'] == model) & 
                             (self.df['platform'] == platform)]
                if not row.empty:
                    values.append(row.iloc[0][metric])
                else:
                    values.append(0)
            
            offset = (i - len(platforms)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=platform, alpha=0.8)
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('.tflite', '').replace('best-', '') 
                            for m in models_to_plot], 
                           rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_speedup_heatmap(self):
        """Crée une heatmap des speedups relatifs"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculer les speedups par rapport au PC
        platforms = sorted(self.df['platform'].unique())
        
        # Prendre les modèles communs à toutes les plateformes
        common_models = set(self.df[self.df['platform'] == platforms[0]]['model_name'])
        for platform in platforms[1:]:
            common_models &= set(self.df[self.df['platform'] == platform]['model_name'])
        
        common_models = sorted(list(common_models))[:15]  # Limiter à 15 modèles
        
        # Créer la matrice de speedup
        speedup_matrix = []
        
        for model in common_models:
            row = []
            baseline = None
            
            for platform in platforms:
                result = self.df[(self.df['model_name'] == model) & 
                               (self.df['platform'] == platform)]
                
                if not result.empty:
                    fps = result.iloc[0]['fps_mean']
                    
                    if baseline is None:
                        baseline = fps
                        row.append(1.0)  # Baseline = 1.0
                    else:
                        speedup = fps / baseline if baseline > 0 else 0
                        row.append(speedup)
                else:
                    row.append(0)
            
            speedup_matrix.append(row)
        
        # Créer la heatmap
        speedup_df = pd.DataFrame(speedup_matrix, 
                                 index=[m.replace('.tflite', '').replace('best-', '') 
                                       for m in common_models],
                                 columns=platforms)
        
        sns.heatmap(speedup_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=1.0, ax=ax, cbar_kws={'label': 'Speedup relatif'})
        
        ax.set_title('Speedup FPS relatif (par rapport à la première plateforme)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Plateforme', fontsize=12)
        ax.set_ylabel('Modèle', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / "speedup_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def plot_efficiency_comparison(self):
        """Compare l'efficacité énergétique (FPS / Watt si disponible)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        platforms = self.df['platform'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(platforms)))
        
        for i, platform in enumerate(platforms):
            platform_data = self.df[self.df['platform'] == platform]
            
            ax.scatter(platform_data['memory_usage_mean'], 
                      platform_data['fps_mean'],
                      s=200, alpha=0.6, c=[colors[i]], 
                      label=platform, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Utilisation mémoire moyenne (MB)', fontsize=12)
        ax.set_ylabel('FPS moyen', fontsize=12)
        ax.set_title('Efficacité: FPS vs Mémoire (par plateforme)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "efficiency_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def generate_comparison_table(self):
        """Génère un tableau de comparaison"""
        # Pour chaque modèle, comparer les métriques entre plateformes
        models = self.df['model_name'].unique()
        platforms = self.df['platform'].unique()
        
        comparison_data = []
        
        for model in models:
            row = {'model': model.replace('.tflite', '').replace('best-', '')}
            
            for platform in platforms:
                result = self.df[(self.df['model_name'] == model) & 
                               (self.df['platform'] == platform)]
                
                if not result.empty:
                    r = result.iloc[0]
                    row[f'{platform}_fps'] = f"{r['fps_mean']:.1f}"
                    row[f'{platform}_inference'] = f"{r['inference_time_mean']:.1f}"
                    row[f'{platform}_memory'] = f"{r['memory_usage_mean']:.0f}"
                else:
                    row[f'{platform}_fps'] = "N/A"
                    row[f'{platform}_inference'] = "N/A"
                    row[f'{platform}_memory'] = "N/A"
            
            comparison_data.append(row)
        
        # Sauvegarder en CSV
        df_comparison = pd.DataFrame(comparison_data)
        csv_path = self.output_dir / "platform_comparison.csv"
        df_comparison.to_csv(csv_path, index=False)
        print(f"✓ Tableau de comparaison sauvegardé: {csv_path}")
        
        return df_comparison
    
    def generate_html_report(self):
        """Génère un rapport HTML de comparaison"""
        platforms = ', '.join(self.df['platform'].unique())
        
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparaison Multi-Plateformes</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
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
        .plot {{
            margin: 20px 0;
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 10px;
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
    </style>
</head>
<body>
    <h1>🔄 Comparaison Multi-Plateformes</h1>
    
    <div class="summary">
        <h2>Plateformes comparées</h2>
        <p><strong>{platforms}</strong></p>
        <p>Nombre total de résultats: {len(self.df)}</p>
    </div>
    
    <h2>📊 Graphiques de comparaison</h2>
    
    <div class="plot">
        <h3>Temps d'inférence par plateforme</h3>
        <img src="inference_comparison.png" alt="Inference comparison">
    </div>
    
    <div class="plot">
        <h3>FPS par plateforme</h3>
        <img src="fps_comparison.png" alt="FPS comparison">
    </div>
    
    <div class="plot">
        <h3>Speedup relatif</h3>
        <img src="speedup_heatmap.png" alt="Speedup heatmap">
    </div>
    
    <div class="plot">
        <h3>Efficacité (FPS vs Mémoire)</h3>
        <img src="efficiency_comparison.png" alt="Efficiency comparison">
    </div>
    
    <footer style="margin-top: 50px; text-align: center; color: #999;">
        <p>Généré automatiquement par compare_platforms.py</p>
    </footer>
</body>
</html>
"""
        
        report_path = self.output_dir / "platform_comparison_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Rapport HTML généré: {report_path}")
    
    def compare(self):
        """Lance toutes les comparaisons"""
        print(f"\n{'='*60}")
        print(f"Comparaison de {len(self.result_files)} fichiers de résultats")
        print(f"Plateformes: {', '.join(self.df['platform'].unique())}")
        print(f"{'='*60}\n")
        
        print("Génération des graphiques de comparaison...")
        
        self.plot_platform_comparison(
            'inference_time_mean', 
            'Temps d\'inférence (ms)',
            'Comparaison des temps d\'inférence entre plateformes',
            'inference_comparison.png'
        )
        
        self.plot_platform_comparison(
            'fps_mean', 
            'FPS',
            'Comparaison des FPS entre plateformes',
            'fps_comparison.png',
            lower_is_better=False
        )
        
        self.plot_speedup_heatmap()
        self.plot_efficiency_comparison()
        
        print("\nGénération du tableau de comparaison...")
        self.generate_comparison_table()
        
        print("\nGénération du rapport HTML...")
        self.generate_html_report()
        
        print(f"\n{'='*60}")
        print(f"Comparaison terminée!")
        print(f"Résultats dans: {self.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare les résultats de benchmarking entre plateformes"
    )
    parser.add_argument('results', nargs='+', 
                       help='Fichiers JSON de résultats à comparer')
    
    args = parser.parse_args()
    
    if len(args.results) < 2:
        print("❌ Au moins 2 fichiers de résultats sont nécessaires pour la comparaison")
        return
    
    comparator = MultiPlatformComparator(args.results)
    comparator.compare()


if __name__ == '__main__':
    main()
