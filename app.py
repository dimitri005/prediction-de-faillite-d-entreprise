from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import base64
import io
import threading

app = Flask(__name__)

# ================================
# Chargement modèle + features
# ================================
model         = joblib.load('models/lightgbm_best_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')
explainer     = shap.TreeExplainer(model)

print("✅ Features chargées :", feature_names)

# ================================
# Verrou pour sécuriser les appels concurrents
# (la comparaison multi-entreprises envoie plusieurs
# requêtes /predict en parallèle ; le modèle et SHAP
# ne sont pas garantis thread-safe sous forte charge)
# ================================
inference_lock = threading.Lock()

# ================================
# Fonction — Graphique SHAP local
# ================================
def generate_shap_chart(shap_vals, feature_values, feature_names):
    # On utilise l'API orientée objet de matplotlib (Figure directe)
    # plutôt que pyplot, qui repose sur un état global non thread-safe.
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()

    colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals]
    y_pos  = np.arange(len(feature_names))

    ax.barh(y_pos, shap_vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n.replace('_', ' ')}\n(val: {v:.3f})"
                        for n, v in zip(feature_names, feature_values)],
                       fontsize=9)
    ax.set_xlabel('Impact SHAP sur la prédiction', fontsize=11)
    ax.set_title('Explication de la Prédiction — SHAP', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    legend = [Patch(color='#e74c3c', label='→ Risque Faillite'),
              Patch(color='#3498db', label='→ Entreprise Saine')]
    ax.legend(handles=legend, loc='lower right')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# ================================
# Routes
# ================================
@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les 10 valeurs du formulaire
        values = []
        for feat in feature_names:
            val = request.form.get(feat, 0)
            values.append(float(val))

        # DataFrame
        X_input = pd.DataFrame([values], columns=feature_names)

        # Prédiction + SHAP — verrouillés pour rester sûrs
        # quand plusieurs requêtes /predict arrivent en parallèle
        # (ex: comparaison de plusieurs entreprises depuis le front-end)
        with inference_lock:
            prediction  = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1] * 100

            shap_values = explainer.shap_values(X_input)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        # Trier par importance absolue
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap'   : sv,
            'value'  : values
        }).assign(abs_shap=lambda x: x['shap'].abs()) \
          .sort_values('abs_shap', ascending=False)

        # Graphique SHAP
        shap_chart = generate_shap_chart(
            shap_df['shap'].tolist(),
            shap_df['value'].tolist(),
            shap_df['feature'].tolist()
        )

        # Niveau de risque
        if probability >= 70:
            niveau_risque  = "ÉLEVÉ"
            couleur_risque = "#e74c3c"
        elif probability >= 40:
            niveau_risque  = "MODÉRÉ"
            couleur_risque = "#f39c12"
        else:
            niveau_risque  = "FAIBLE"
            couleur_risque = "#27ae60"

        return jsonify({
            'prediction'    : int(prediction),
            'probabilite'   : round(probability, 2),
            'niveau_risque' : niveau_risque,
            'couleur_risque': couleur_risque,
            'shap_chart'    : shap_chart,
            'verdict'       : 'Faillite probable' if prediction == 1 else 'Entreprise saine',
            'top_features'  : [{'name': row['feature'], 'shap': round(row['shap'], 4)}
                                for _, row in shap_df.iterrows()]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # threaded=True : permet de traiter en parallèle les requêtes /predict
    # envoyées simultanément par la comparaison multi-entreprises
    app.run(debug=True, threaded=True)