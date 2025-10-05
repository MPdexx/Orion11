import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class OrionIIVisualizer:
    def __init__(self, data):
        self.data = data
    
    def show_basic_stats(self):
        """Mostrar estadísticas básicas del dataset"""
        if self.data is None:
            print("❌ No hay datos para mostrar")
            return
        
        print("\n" + "="*60)
        print("📈 ESTADÍSTICAS BÁSICAS DE ORION-11")
        print("="*60)
        print(f"🪐 Total de objetos KOI: {len(self.data):,}")
        print(f"📋 Columnas disponibles: {len(self.data.columns)}")
        
        if 'koi_disposition' in self.data.columns:
            print("\n🎯 Distribución de disposiciones KOI:")
            disp_stats = self.data['koi_disposition'].value_counts()
            for disposition, count in disp_stats.items():
                percentage = (count / len(self.data)) * 100
                print(f"   {disposition}: {count:,} ({percentage:.1f}%)")
        
        if 'koi_score' in self.data.columns:
            print("\n📊 Estadísticas de KOI Score:")
            score_stats = self.data['koi_score'].describe()
            print(f"   Mínimo: {score_stats['min']:.3f}")
            print(f"   Máximo: {score_stats['max']:.3f}")
            print(f"   Media: {score_stats['mean']:.3f}")
            print(f"   Mediana: {self.data['koi_score'].median():.3f}")
    
    def create_visualizations(self):
        """Crear visualizaciones básicas"""
        if self.data is None:
            print("❌ No hay datos para visualizar")
            return
        
        print("\n🎨 Generando visualizaciones...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Exploratorio - Datos KOI (Orion-11)', fontsize=16, fontweight='bold')
        
        # 1. Distribución de disposiciones
        if 'koi_disposition' in self.data.columns:
            disposition_counts = self.data['koi_disposition'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(disposition_counts)))
            axes[0,0].pie(disposition_counts.values, labels=disposition_counts.index, 
                         autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,0].set_title('Distribución de Disposiciones KOI', fontweight='bold')
        
        # 2. Histograma de koi_score
        if 'koi_score' in self.data.columns:
            axes[0,1].hist(self.data['koi_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_xlabel('KOI Score')
            axes[0,1].set_ylabel('Frecuencia')
            axes[0,1].set_title('Distribución de KOI Scores', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Período vs Radio
        if all(col in self.data.columns for col in ['koi_period', 'koi_prad', 'koi_score']):
            scatter = axes[1,0].scatter(self.data['koi_period'], self.data['koi_prad'], 
                                       c=self.data['koi_score'], cmap='viridis', alpha=0.6, s=20)
            axes[1,0].set_xlabel('Período Orbital (días)')
            axes[1,0].set_ylabel('Radio Planetario (Radios Terrestres)')
            axes[1,0].set_title('Período vs Radio (coloreado por Score)', fontweight='bold')
            axes[1,0].set_yscale('log')
            axes[1,0].set_xscale('log')
            plt.colorbar(scatter, ax=axes[1,0], label='KOI Score')
        
        # 4. Temperatura vs Insolación
        if all(col in self.data.columns for col in ['koi_teq', 'koi_insol', 'koi_score']):
            scatter = axes[1,1].scatter(self.data['koi_teq'], self.data['koi_insol'], 
                                       c=self.data['koi_score'], cmap='plasma', alpha=0.6, s=20)
            axes[1,1].set_xlabel('Temperatura de Equilibrio (K)')
            axes[1,1].set_ylabel('Insolación (Tierra = 1)')
            axes[1,1].set_title('Temperatura vs Insolación', fontweight='bold')
            axes[1,1].set_yscale('log')
            plt.colorbar(scatter, ax=axes[1,1], label='KOI Score')
        
        plt.tight_layout()
        plt.show()