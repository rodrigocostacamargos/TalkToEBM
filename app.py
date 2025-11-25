from flask import Flask, render_template, request, jsonify, Response
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
        t2ebm, X_test, y_test, feature_descriptions, get_feature_description
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
    # Passar features com descrições para o template
    features_with_desc = []
    if MODEL_LOADED:
        for fname in feature_names:
            features_with_desc.append({
                'name': fname,
                'description': get_feature_description(fname)
            })
    
    return render_template('index.html', 
                         feature_names=feature_names if MODEL_LOADED else [],
                         features_with_desc=features_with_desc,
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
    """API para listar as features disponíveis com descrições"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    features = []
    for i, name in enumerate(feature_names):
        features.append({
            'index': i,
            'name': name,
            'description': get_feature_description(name),
            'type': 'categorical' if 'Curso' in name or 'Semestre' in name else 'numeric'
        })
    
    return jsonify(features)

@app.route('/api/feature_description/<string:feature_name>')
def get_feature_desc(feature_name):
    """API para obter a descrição de uma feature específica"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    description = get_feature_description(feature_name)
    return jsonify({
        'feature_name': feature_name,
        'description': description
    })

@app.route('/api/visualize_global')
def visualize_global():
    """API para visualização global do modelo EBM usando visualize()"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        ebm_global = ebm.explain_global()
        # Usar visualize() para obter o objeto plotly e converter para HTML
        fig = ebm_global.visualize()
        if hasattr(fig, 'to_html'):
            # Se for um objeto plotly
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        else:
            # Fallback: tentar obter dados do objeto
            html_content = f"<html><body><pre>{str(ebm_global.data())}</pre></body></html>"
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/visualize_feature/<int:feature_index>')
def visualize_feature(feature_index):
    """API para visualização de uma feature específica por índice usando visualize()"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        ebm_global = ebm.explain_global()
        # Para visualização de feature específica
        if feature_index < len(feature_names):
            fig = ebm_global.visualize(feature_index)
            if hasattr(fig, 'to_html'):
                html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
            else:
                html_content = f"<html><body><pre>{str(ebm_global.data(feature_index))}</pre></body></html>"
            return Response(html_content, mimetype='text/html')
        else:
            return jsonify({'error': 'Índice de feature inválido'}), 400
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/visualize_feature_by_name/<string:feature_name>')
def visualize_feature_by_name(feature_name):
    """API para visualização de uma feature específica pelo nome usando visualize()"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        # Encontrar o índice da feature pelo nome
        if feature_name in feature_names:
            feature_index = feature_names.index(feature_name)
        else:
            return jsonify({'error': f'Feature "{feature_name}" não encontrada. Features disponíveis: {feature_names}'}), 400
        
        ebm_global = ebm.explain_global()
        # Usar o índice para visualizar (visualize() aceita apenas índices inteiros)
        fig = ebm_global.visualize(feature_index)
        if hasattr(fig, 'to_html'):
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        else:
            html_content = f"<html><body><pre>{str(ebm_global.data(feature_index))}</pre></body></html>"
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/visualize_local/<int:sample_index>')
def visualize_local(sample_index):
    """API para visualização de explicação local usando visualize()"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Modelo não carregado'}), 500
    
    try:
        # Explicação local para os primeiros 5 exemplos
        ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
        if sample_index < 5:
            fig = ebm_local.visualize(sample_index)
            if hasattr(fig, 'to_html'):
                html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
            else:
                html_content = f"<html><body><pre>{str(ebm_local.data(sample_index))}</pre></body></html>"
            return Response(html_content, mimetype='text/html')
        else:
            return jsonify({'error': 'Índice de amostra inválido'}), 400
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

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
