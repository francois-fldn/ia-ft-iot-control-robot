#!/usr/bin/env python3
"""
Script d'analyse et visualisation des resultats de benchmarking
- Graphiques separes pour modeles standard et prunes
- Code couleur: TFLite vs ONNX
- Metriques: temps d'inference, FPS, memoire, confiance, FPS/Watt
- Comparaison Pi4 vs Coral
- Scatter plots memoire/FPS et temps d'inference
- Consommation electrique avec soustraction de la consommation a vide
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path
import argparse


# Configuration des couleurs par type de runtime
COULEURS = {
    'tflite': '#2196F3',  # Bleu
    'onnx': '#FF9800',    # Orange
    'edgetpu': '#4CAF50'  # Vert pour Coral
}

# Couleurs par plateforme
COULEURS_PLATEFORME = {
    'raspberry_pi4': '#2196F3',      # Bleu
    'raspberry_pi4_coral': '#4CAF50', # Vert
    'Pi4': '#2196F3',
    'Coral': '#4CAF50'
}

# Tension pour le calcul de puissance (5V pour Pi)
TENSION = 5.0

# Modeles selectionnes pour la comparaison Pi4 vs Coral
MODELES_PI4_COMPARAISON = [
    'best-fp16_256_pruned.onnx', 'best-fp32_256_pruned.onnx',
    'best-int8_320.tflite', 'best-int8_256.tflite',
    'best-int8_320_pruned.tflite', 'best-int8_256_pruned.tflite'
]

MODELES_CORAL = [
    'best-int8_edgetpu256.tflite', 'best-int8_edgetpu320.tflite',
    'best-int8_256_pruned_edgetpu.tflite', 'best-int8_320_pruned_edgetpu.tflite'
]


def charger_tous_les_runs(dossier: str) -> pd.DataFrame:
    # Charge tous les fichiers CSV de benchmark d'un dossier
    fichiers_csv = sorted(glob.glob(os.path.join(dossier, "benchmark_results_*.csv")))
    
    toutes_donnees = []
    for i, fichier_csv in enumerate(fichiers_csv):
        df = pd.read_csv(fichier_csv)
        df['run_id'] = i + 1
        toutes_donnees.append(df)
    
    if not toutes_donnees:
        return pd.DataFrame()
    
    return pd.concat(toutes_donnees, ignore_index=True)


def normaliser_nom_modele(nom: str) -> str:
    # Normalise le nom du modele pour gerer les inconsistances best_ vs best-
    if nom.startswith('best_'):
        nom = 'best-' + nom[5:]
    return nom


def charger_consommation(chemin_json: str) -> tuple:
    # Charge les donnees de consommation et retourne un dict avec model_name comme cle
    if not os.path.exists(chemin_json):
        return {}, 0.0
        
    with open(chemin_json, 'r') as f:
        donnees = json.load(f)
    
    dict_puissance = {}
    puissance_idle = 0.0
    
    for entree in donnees:
        nom_modele = entree.get('model_name', '')
        if nom_modele == 'None':
            # Consommation a vide - utiliser moyenne de min/max
            puissance_idle = (entree.get('conso_ampere_max', 0) + entree.get('conso_ampere_min', 0)) / 2
        else:
            # Stocker avec nom normalise comme cle
            dict_puissance[normaliser_nom_modele(nom_modele)] = entree.get('conso_ampere_mean', 0)
    
    return dict_puissance, puissance_idle


def obtenir_nom_court(nom_modele: str) -> str:
    # Extrait le nom court du fichier modele
    nom = nom_modele.replace('best-', '').replace('best_', '')
    nom = nom.replace('.tflite', '').replace('.onnx', '')
    return nom


def calculer_stats_par_modele(df: pd.DataFrame) -> pd.DataFrame:
    # Calcule moyenne et ecart-type sur les runs pour chaque modele
    metriques = ['inference_time_mean', 'fps_mean', 'memory_usage_mean', 
                 'max_confidence_mean', 'total_time_mean']
    
    groupe = df.groupby('model_name').agg({
        **{m: ['mean', 'std'] for m in metriques},
        'runtime': 'first',
        'model_type': 'first',
        'input_size': 'first',
        'is_edgetpu': 'first'
    }).reset_index()
    
    # Aplatir les noms de colonnes
    groupe.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in groupe.columns]
    
    return groupe


def ajouter_donnees_puissance(stats: pd.DataFrame, dict_puissance: dict, 
                               puissance_idle: float) -> pd.DataFrame:
    # Ajoute les colonnes de puissance et FPS/Watt au DataFrame
    stats = stats.copy()
    stats['puissance_amps'] = stats['model_name'].apply(
        lambda x: dict_puissance.get(normaliser_nom_modele(x), np.nan))
    stats['puissance_nette_amps'] = stats['puissance_amps'] - puissance_idle
    stats['puissance_watts'] = stats['puissance_nette_amps'] * TENSION
    stats['fps_par_watt'] = stats['fps_mean_mean'] / stats['puissance_watts']
    stats['nom_court'] = stats['model_name'].apply(obtenir_nom_court)
    stats['type_runtime'] = stats.apply(
        lambda r: 'edgetpu' if r['is_edgetpu_first'] else r['runtime_first'], axis=1)
    return stats


def tracer_metriques_plateforme(stats: pd.DataFrame, dossier_sortie: str, 
                                 prefixe: str = "pi4"):
    # Genere les graphiques avec separation standard/prunes pour une plateforme
    
    # Separer standard et prune
    prunes = stats[stats['model_type_first'] == 'pruned'].copy()
    standard = stats[stats['model_type_first'] == 'standard'].copy()
    
    # Trier par FPS
    prunes = prunes.sort_values('fps_mean_mean', ascending=True)
    standard = standard.sort_values('fps_mean_mean', ascending=True)
    
    # Metriques a tracer
    config_metriques = [
        ('inference_time_mean_mean', 'inference_time_mean_std', 
         'Temps d\'inference (ms)', 'inference_time'),
        ('fps_mean_mean', 'fps_mean_std', 'FPS', 'fps'),
        ('memory_usage_mean_mean', 'memory_usage_mean_std', 
         'Utilisation memoire (MB)', 'memory'),
        ('max_confidence_mean_mean', 'max_confidence_mean_std', 
         'Score de confiance', 'confidence'),
        ('fps_par_watt', None, 'FPS/Watt (Puissance nette)', 'fps_per_watt'),
    ]
    
    for donnees, suffixe in [(standard, 'standard'), (prunes, 'pruned')]:
        if donnees.empty:
            continue
            
        for col_moyenne, col_std, label_y, nom_fichier in config_metriques:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            pos_y = np.arange(len(donnees))
            couleurs = [COULEURS.get(rt, '#888888') for rt in donnees['type_runtime']]
            
            barres = ax.barh(pos_y, donnees[col_moyenne], color=couleurs, alpha=0.8, height=0.6)
            
            # Barres d'erreur
            if col_std and col_std in donnees.columns:
                ax.errorbar(donnees[col_moyenne], pos_y, xerr=donnees[col_std], 
                           fmt='none', color='black', capsize=3, capthick=1)
            
            ax.set_yticks(pos_y)
            ax.set_yticklabels(donnees['nom_court'], fontsize=10)
            ax.set_xlabel(label_y, fontsize=12)
            
            titre_type = "Prunes" if suffixe == "pruned" else "Standard"
            ax.set_title(f'{label_y} (Modeles {titre_type})', fontsize=14, fontweight='bold')
            
            # Valeurs sur les barres
            for barre, val in zip(barres, donnees[col_moyenne]):
                if not np.isnan(val):
                    ax.text(val + ax.get_xlim()[1] * 0.01, barre.get_y() + barre.get_height()/2,
                           f'{val:.2f}', va='center', fontsize=9)
            
            # Legende
            from matplotlib.patches import Patch
            elements_legende = [
                Patch(facecolor=COULEURS['tflite'], label='TFLite'),
                Patch(facecolor=COULEURS['onnx'], label='ONNX')
            ]
            ax.legend(handles=elements_legende, loc='lower right')
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dossier_sortie, f'{prefixe}_{nom_fichier}_{suffixe}.png'), dpi=150)
            plt.close()
            print(f"Sauvegarde: {prefixe}_{nom_fichier}_{suffixe}.png")


def tracer_comparaison_coral(stats_pi4: pd.DataFrame, stats_coral: pd.DataFrame,
                              dossier_sortie: str):
    # Compare Pi4 et Coral sur les modeles selectionnes
    
    # Filtrer les modeles de comparaison
    pi4_filtre = stats_pi4[stats_pi4['model_name'].isin(MODELES_PI4_COMPARAISON)].copy()
    coral_filtre = stats_coral[stats_coral['model_name'].isin(MODELES_CORAL)].copy()
    
    if pi4_filtre.empty and coral_filtre.empty:
        print("Aucun modele de comparaison trouve")
        return
    
    # Ajouter la plateforme
    pi4_filtre['plateforme'] = 'Pi4'
    coral_filtre['plateforme'] = 'Coral'
    
    # Combiner
    combine = pd.concat([pi4_filtre, coral_filtre], ignore_index=True)
    combine = combine.sort_values('fps_mean_mean', ascending=True)
    
    # Metriques de comparaison
    config_metriques = [
        ('inference_time_mean_mean', 'inference_time_mean_std', 
         'Temps d\'inference (ms)', 'comparison_inference_time'),
        ('fps_mean_mean', 'fps_mean_std', 'FPS', 'comparison_fps'),
        ('memory_usage_mean_mean', 'memory_usage_mean_std', 
         'Utilisation memoire (MB)', 'comparison_memory'),
        ('max_confidence_mean_mean', 'max_confidence_mean_std', 
         'Score de confiance', 'comparison_confidence'),
        ('fps_par_watt', None, 'FPS/Watt (Puissance nette)', 'comparison_fps_per_watt'),
    ]
    
    for col_moyenne, col_std, label_y, nom_fichier in config_metriques:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pos_y = np.arange(len(combine))
        couleurs = [COULEURS_PLATEFORME.get(p, '#888888') for p in combine['plateforme']]
        
        # Labels avec plateforme
        labels = [f"{row['nom_court']}\n({row['plateforme']})" for _, row in combine.iterrows()]
        
        barres = ax.barh(pos_y, combine[col_moyenne], color=couleurs, alpha=0.8, height=0.7)
        
        # Barres d'erreur
        if col_std and col_std in combine.columns:
            ax.errorbar(combine[col_moyenne], pos_y, xerr=combine[col_std], 
                       fmt='none', color='black', capsize=3, capthick=1)
        
        ax.set_yticks(pos_y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(label_y, fontsize=12)
        ax.set_title(f'Pi4 vs Coral - {label_y}', fontsize=14, fontweight='bold')
        
        # Valeurs
        for barre, val in zip(barres, combine[col_moyenne]):
            if not np.isnan(val):
                ax.text(val + ax.get_xlim()[1] * 0.01, barre.get_y() + barre.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=9)
        
        # Legende
        from matplotlib.patches import Patch
        elements_legende = [
            Patch(facecolor=COULEURS_PLATEFORME['Pi4'], label='Pi4 (CPU)'),
            Patch(facecolor=COULEURS_PLATEFORME['Coral'], label='Pi4 + Coral TPU')
        ]
        ax.legend(handles=elements_legende, loc='lower right')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_sortie, f'{nom_fichier}.png'), dpi=150)
        plt.close()
        print(f"Sauvegarde: {nom_fichier}.png")


def tracer_scatter_efficiency(stats: pd.DataFrame, dossier_sortie: str, suffixe: str = "pi4"):
    # Scatter plot: Memoire vs FPS avec couleur = temps d'inference
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        stats['memory_usage_mean_mean'], 
        stats['fps_mean_mean'],
        s=200, alpha=0.7, 
        c=stats['inference_time_mean_mean'],
        cmap='viridis'
    )
    
    # Annotations
    for _, row in stats.iterrows():
        ax.annotate(
            row['nom_court'], 
            (row['memory_usage_mean_mean'], row['fps_mean_mean']),
            fontsize=7, ha='center', va='bottom'
        )
    
    ax.set_xlabel('Utilisation memoire moyenne (MB)', fontsize=12)
    ax.set_ylabel('FPS moyen', fontsize=12)
    ax.set_title('Efficacite: FPS vs Memoire', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temps d\'inference (ms)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_sortie, f'scatter_efficiency_{suffixe}.png'), dpi=150)
    plt.close()
    print(f"Sauvegarde: scatter_efficiency_{suffixe}.png")


def tracer_scatter_comparaison(stats_pi4: pd.DataFrame, stats_coral: pd.DataFrame,
                                dossier_sortie: str):
    # Scatter plot pour les modeles de comparaison Pi4 vs Coral
    
    # Filtrer
    pi4_filtre = stats_pi4[stats_pi4['model_name'].isin(MODELES_PI4_COMPARAISON)].copy()
    coral_filtre = stats_coral[stats_coral['model_name'].isin(MODELES_CORAL)].copy()
    
    if pi4_filtre.empty and coral_filtre.empty:
        return
    
    pi4_filtre['plateforme'] = 'Pi4'
    coral_filtre['plateforme'] = 'Coral'
    
    combine = pd.concat([pi4_filtre, coral_filtre], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Couleurs par plateforme
    couleurs = [COULEURS_PLATEFORME.get(p, '#888888') for p in combine['plateforme']]
    
    scatter = ax.scatter(
        combine['memory_usage_mean_mean'], 
        combine['fps_mean_mean'],
        s=200, alpha=0.7, 
        c=couleurs
    )
    
    # Annotations
    for _, row in combine.iterrows():
        ax.annotate(
            f"{row['nom_court']}", 
            (row['memory_usage_mean_mean'], row['fps_mean_mean']),
            fontsize=7, ha='center', va='bottom'
        )
    
    ax.set_xlabel('Utilisation memoire moyenne (MB)', fontsize=12)
    ax.set_ylabel('FPS moyen', fontsize=12)
    ax.set_title('Comparaison Pi4 vs Coral: FPS vs Memoire', fontsize=14, fontweight='bold')
    
    # Legende
    from matplotlib.patches import Patch
    elements_legende = [
        Patch(facecolor=COULEURS_PLATEFORME['Pi4'], label='Pi4 (CPU)'),
        Patch(facecolor=COULEURS_PLATEFORME['Coral'], label='Pi4 + Coral TPU')
    ]
    ax.legend(handles=elements_legende, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dossier_sortie, 'scatter_comparison.png'), dpi=150)
    plt.close()
    print(f"Sauvegarde: scatter_comparison.png")


class AnalyseurBenchmark:
    # Analyseur de resultats de benchmarking avec comparaison multi-plateforme
    
    def __init__(self, dossier_base: str):
        # Args: dossier_base: Dossier contenant benchmark_results_pi4 et benchmark_results_coral
        self.dossier_base = Path(dossier_base)
        self.dossier_sortie = self.dossier_base / "plots"
        self.dossier_sortie.mkdir(exist_ok=True)
        
        # Chemins des sous-dossiers
        self.dossier_pi4 = self.dossier_base / "benchmark_results_pi4"
        self.dossier_coral = self.dossier_base / "benchmark_results_coral"
        
        # Charger les donnees Pi4
        self.df_pi4 = pd.DataFrame()
        self.stats_pi4 = pd.DataFrame()
        self.puissance_pi4 = {}
        self.idle_pi4 = 0.0
        
        if self.dossier_pi4.exists():
            print("Chargement des donnees Pi4...")
            self.df_pi4 = charger_tous_les_runs(str(self.dossier_pi4))
            if not self.df_pi4.empty:
                print(f"  Charge {len(self.df_pi4)} lignes de {self.df_pi4['run_id'].nunique()} runs")
                self.puissance_pi4, self.idle_pi4 = charger_consommation(
                    str(self.dossier_pi4 / "benchmark_conso_Amp.json"))
                if self.idle_pi4 > 0:
                    print(f"  Consommation idle: {self.idle_pi4:.3f}A ({self.idle_pi4 * TENSION:.2f}W)")
                self.stats_pi4 = calculer_stats_par_modele(self.df_pi4)
                self.stats_pi4 = ajouter_donnees_puissance(self.stats_pi4, self.puissance_pi4, self.idle_pi4)
        
        # Charger les donnees Coral
        self.df_coral = pd.DataFrame()
        self.stats_coral = pd.DataFrame()
        self.puissance_coral = {}
        self.idle_coral = 0.0
        
        if self.dossier_coral.exists():
            print("Chargement des donnees Coral...")
            self.df_coral = charger_tous_les_runs(str(self.dossier_coral))
            if not self.df_coral.empty:
                print(f"  Charge {len(self.df_coral)} lignes de {self.df_coral['run_id'].nunique()} runs")
                self.puissance_coral, self.idle_coral = charger_consommation(
                    str(self.dossier_coral / "benchmark_conso_Amp.json"))
                if self.idle_coral > 0:
                    print(f"  Consommation idle: {self.idle_coral:.3f}A ({self.idle_coral * TENSION:.2f}W)")
                self.stats_coral = calculer_stats_par_modele(self.df_coral)
                self.stats_coral = ajouter_donnees_puissance(self.stats_coral, self.puissance_coral, self.idle_coral)
    
    def analyser(self):
        # Lance toutes les analyses et genere les graphiques
        print(f"\n{'='*60}")
        print(f"Analyse des resultats de benchmarking")
        print(f"{'='*60}\n")
        
        # Graphiques Pi4
        if not self.stats_pi4.empty:
            print("--- Graphiques Pi4 ---")
            tracer_metriques_plateforme(self.stats_pi4, str(self.dossier_sortie), "pi4")
            tracer_scatter_efficiency(self.stats_pi4, str(self.dossier_sortie), "pi4_all")
        
        # Comparaison Pi4 vs Coral
        if not self.stats_pi4.empty and not self.stats_coral.empty:
            print("\n--- Comparaison Pi4 vs Coral ---")
            tracer_comparaison_coral(self.stats_pi4, self.stats_coral, str(self.dossier_sortie))
            tracer_scatter_comparaison(self.stats_pi4, self.stats_coral, str(self.dossier_sortie))
        
        print(f"\n{'='*60}")
        print(f"Analyse terminee!")
        print(f"Graphiques dans: {self.dossier_sortie}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyse des resultats de benchmarking")
    parser.add_argument('results', type=str, 
                       help='Dossier benchmark_results contenant benchmark_results_pi4 et benchmark_results_coral')
    
    args = parser.parse_args()
    
    analyseur = AnalyseurBenchmark(args.results)
    analyseur.analyser()


if __name__ == '__main__':
    main()
