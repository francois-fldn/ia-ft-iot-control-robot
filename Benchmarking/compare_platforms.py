#!/usr/bin/env python3
"""
Script de comparaison multi-plateformes - Version Simplifiée
Compare tous les modèles de chaque plateforme côte à côte
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


def load_aggregated_data(directory):
    """Charge et agrège les données d'un dossier/fichier"""
    source = Path(directory)
    all_data = []
    
    # Charger les résultats
    if source.is_dir():
        files = list(source.glob("benchmark_results_*.json"))
        for f in files:
            with open(f, 'r') as jf:
                all_data.extend(json.load(jf))
    elif source.suffix == '.json':
        with open(source, 'r') as f:
            all_data = json.load(f)
    else:
        return pd.DataFrame(), {}
    
    if not all_data:
        return pd.DataFrame(), {}
    
    # Agréger par modèle (moyenne et écart-type des répétitions)
    df = pd.DataFrame(all_data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = ['model_name', 'platform', 'model_type', 'input_size', 'is_edgetpu', 'timestamp']
    numeric_cols = [c for c in numeric_cols if c not in meta_cols]
    
    # Calculer uniquement moyenne
    df_agg = df.groupby('model_name')[numeric_cols].mean().reset_index()
    
    # Récupérer métadonnées
    meta_cols_existing = [c for c in meta_cols if c in df.columns]
    df_meta = df[meta_cols_existing].drop_duplicates(subset=['model_name'])
    
    # Merge
    df_final = pd.merge(df_meta, df_agg, on='model_name', how='inner')
    
    # Charger consommation si disponible
    conso_data = {}
    baseline_power = 0.0  # Consommation à vide
    
    if source.is_dir():
        conso_file = source / "benchmark_conso_Amp.json"
    else:
        conso_file = source.parent / "benchmark_conso_Amp.json"
    
    if conso_file.exists():
        with open(conso_file, 'r') as f:
            conso_list = json.load(f)
            
            # D'abord, trouver la consommation à vide (baseline)
            for item in conso_list:
                if item.get('model_name') == 'None' or item.get('model_name') is None:
                    # Consommation idle - utiliser moyenne si disponible, sinon max
                    baseline_power = item.get('conso_ampere_mean', 
                                             item.get('conso_ampere_max', 0)) * 5.0
                    print(f"  Baseline idle: {baseline_power:.2f}W")
                    break
            
            # Ensuite, charger les consommations des modèles et soustraire le baseline
            for item in conso_list:
                model_name = item.get('model_name')
                if model_name and model_name != 'None':
                    conso_ampere = item.get('conso_ampere_mean', 0)
                    total_power = conso_ampere * 5.0  # Watts
                    net_power = max(0, total_power - baseline_power)  # Soustraire baseline
                    
                    # Enregistrer avec le nom original ET une version normalisée (- → _)
                    conso_data[model_name] = net_power
                    # Ajouter aussi la version avec underscore pour compatibilité
                    normalized_name = model_name.replace('-', '_')
                    if normalized_name != model_name:
                        conso_data[normalized_name] = net_power
    
    return df_final, conso_data


class PlatformComparator:
    """Compare les résultats entre plateformes"""
    
    def __init__(self, platform_sources):
        """
        Args:
            platform_sources: Liste des chemins vers fichiers/dossiers de résultats
        """
        self.output_dir = Path("benchmark_results/comparison")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Charger toutes les données
        all_dfs = []
        all_conso = {}
        
        for source in platform_sources:
            df, conso = load_aggregated_data(source)
            if not df.empty:
                platform_name = df['platform'].iloc[0]
                
                # Ajouter consommation au DataFrame
                df['power_w'] = df['model_name'].apply(lambda x: conso.get(x, 0))
                df['efficiency'] = df.apply(
                    lambda row: row['fps_mean'] / row['power_w'] if row['power_w'] > 0 else 0,
                    axis=1
                )
                
                # Label court pour affichage
                df['label'] = df['model_name'].apply(
                    lambda x: x.replace('best-', '').replace('.onnx', '').replace('.tflite', '')
                              .replace('_edgetpu', '-TPU')[:30]  # Limite 30 chars
                )
                
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("Aucune donnée chargée")
        
        # Combiner toutes les plateformes
        self.df = pd.concat(all_dfs, ignore_index=True)
        
        # Couleurs par plateforme
        self.platforms = self.df['platform'].unique()
        self.colors = {
            'raspberry_pi4': '#1f77b4',  # Bleu
            'raspberry_pi4_coral': '#ff7f0e',  # Orange
            'jetson_orin': '#d62728',  # Rouge
            'pc': '#2ca02c',  # Vert
        }
        
        sns.set_style("whitegrid")
    
    def plot_metric(self, metric, title, ylabel, filename):
        """Crée un graphique comparatif pour une métrique"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(self.df))
        
        for platform in self.platforms:
            mask = self.df['platform'] == platform
            indices = np.where(mask)[0]
            values = self.df.loc[mask, metric].values
            
            color = self.colors.get(platform, 'gray')
            ax.bar(indices, values, label=platform, alpha=0.8, color=color)
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.df['label'], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sauvegarde: {filename}")
        plt.close()
    
    def generate_csv(self):
        """Génère un CSV récapitulatif"""
        summary = self.df[['label', 'platform', 'fps_mean', 'inference_time_mean', 
                          'memory_usage_mean', 'power_w', 'efficiency']].copy()
        
        summary.columns = ['Modèle', 'Plateforme', 'FPS', 'Inférence (ms)', 
                           'Mémoire (MB)', 'Puissance (W)', 'Efficacité (FPS/W)']
        
        summary = summary.round({
            'FPS': 1,
            'Inférence (ms)': 1,
            'Mémoire (MB)': 0,
            'Puissance (W)': 2,
            'Efficacité (FPS/W)': 1
        })
        
        csv_path = self.output_dir / "platform_comparison.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Sauvegarde: platform_comparison.csv")
        
        return summary
    
    def compare(self):
        """Lance toutes les comparaisons"""
        print(f"\n{'='*60}")
        print(f"Comparaison de {len(self.platforms)} plateforme(s)")
        print(f"Plateformes: {', '.join(self.platforms)}")
        print(f"Total modèles: {len(self.df)}")
        print(f"{'='*60}\n")
        
        print("Generation des graphiques...")
        
        self.plot_metric('inference_time_mean', 
                        'Comparaison des Temps d\'Inférence', 
                        'Temps (ms)',
                        'inference_comparison.png')
        
        self.plot_metric('fps_mean',
                        'Comparaison des FPS',
                        'FPS',
                        'fps_comparison.png')
        
        self.plot_metric('memory_usage_mean',
                        'Comparaison de la Mémoire',
                        'Mémoire (MB)',
                        'memory_comparison.png')
        
        self.plot_metric('power_w',
                        'Comparaison de la Consommation Électrique',
                        'Puissance (Watts)',
                        'power_comparison.png')
        
        self.plot_metric('efficiency',
                        'Comparaison de l\'Efficacité Énergétique',
                        'FPS / Watt',
                        'efficiency_comparison.png')
        
        print("\n📋 Génération du CSV...")
        summary = self.generate_csv()
        
        print(f"\n{'='*60}")
        print("Comparaison terminee!")
        print(f"Résultats dans: {self.output_dir}")
        print(f"{'='*60}")
        
        print("\nResume (10 premiers modeles):")
        print(summary.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compare les résultats de benchmarking entre plateformes"
    )
    parser.add_argument('results', nargs='+',
                       help='Fichiers JSON ou dossiers de résultats à comparer')
    
    args = parser.parse_args()
    
    comparator = PlatformComparator(args.results)
    comparator.compare()


if __name__ == '__main__':
    main()
