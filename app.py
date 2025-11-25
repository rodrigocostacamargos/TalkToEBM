from flask import Flask, render_template, request, jsonify
import sys
import os
import subprocess
import json
import tempfile

app = Flask(__name__)

# Adicionar o diretório atual ao path para importar o módulo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar funções do nosso script
try:
    from evasao_upe import (
        ebm, feature_names, dataset_description, y_axis_description,
        t2ebm
    )
    import t2ebm.graphs as graphs
    import textwrap
    
    # Monkey patch para substituir a função chat_completion do t2ebm.llm
    from deepseek_llm import chat_completion as deepseek_chat_completion
    t2ebm.llm.chat_completion = deepseek_chat_completion
    
    MODEL_LOADED = True
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    MODEL_LOADED = False

@app.route('/')
def index():
    """Página inicial da aplicação"""
    return render_template('index.html', 
                         feature_names=feature_names if MODEL_LOADED else [],
                         model_loaded=MODEL_LOADED)

@app.route('/api/describe_graph', methods=['POST'])
def describe_graph():
    """API para descrever um gráfico específico"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    data = request.get_json()
    feature_index = data.get('feature_index', 0)
    custom_prompt = data.get('custom_prompt', '')
    language = data.get('language', 'Portuguese (Brazil)')
    model = data.get('model', 'deepseek-chat')
    
    try:
        # Usar prompt customizado se fornecido
        if custom_prompt:
            graph = graphs.extract_graph(ebm, feature_index)
            graph_as_text = graphs.graph_to_text(graph, max_tokens=1000)
            
            # Construir prompt personalizado
            prompt = t2ebm.prompts.describe_graph(
                graph_as_text,
                graph_description=y_axis_description,
                dataset_description=dataset_description,
                task_description=custom_prompt,
                language=language
            )
            
            # Usar o modelo selecionado para análise
            description = t2ebm.describe_graph(
                model, 
                ebm, 
                feature_index,
                graph_description=y_axis_description,
                dataset_description=dataset_description,
                language=language
            )
        else:
            # Usar o modelo selecionado para análise
            description = t2ebm.describe_graph(
                model, 
                ebm, 
                feature_index,
                graph_description=y_axis_description,
                dataset_description=dataset_description,
                language=language
            )
        
        return jsonify({
            'success': True,
            'description': description,
            'feature_name': feature_names[feature_index] if feature_index < len(feature_names) else f'Feature {feature_index}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/describe_model', methods=['POST'])
def describe_model():
    """API para descrever o modelo completo"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    data = request.get_json()
    custom_prompt = data.get('custom_prompt', '')
    language = data.get('language', 'Portuguese (Brazil)')
    model = data.get('model', 'deepseek-chat')
    
    try:
        # Usar o modelo selecionado para análise do modelo completo
        description = t2ebm.describe_ebm(
            model, 
            ebm,
            dataset_description=dataset_description,
            outcome_description=y_axis_description,
            language=language
        )
        
        return jsonify({
            'success': True,
            'description': description
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features')
def get_features():
    """API para listar as features disponíveis"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    features = []
    for i, name in enumerate(feature_names):
        features.append({
            'index': i,
            'name': name,
            'type': 'categorical' if 'Curso' in name or 'Semestre' in name else 'numeric'
        })
    
    return jsonify(features)

@app.route('/api/health')
def health_check():
    """Endpoint de health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'features_count': len(feature_names) if MODEL_LOADED else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
