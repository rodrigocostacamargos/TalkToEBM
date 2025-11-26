#!/bin/bash
# Script para configurar variáveis de ambiente para o TalkToEBM
# Execute: source setup_env.sh

echo "=== Configuração de Variáveis de Ambiente TalkToEBM ==="

# Verificar se as variáveis já estão configuradas
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY já configurada"
else
    echo "⚠️  OPENAI_API_KEY não configurada"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ ANTHROPIC_API_KEY já configurada"
else
    echo "⚠️  ANTHROPIC_API_KEY não configurada"
fi

if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo "✅ DEEPSEEK_API_KEY já configurada"
else
    echo "⚠️  DEEPSEEK_API_KEY não configurada"
fi

echo ""
echo "Para configurar permanentemente, adicione as seguintes linhas ao seu ~/.bashrc:"
echo ""
echo '# TalkToEBM API Keys'
echo 'export OPENAI_API_KEY="sua-chave-openai"'
echo 'export ANTHROPIC_API_KEY="sua-chave-anthropic"'
echo 'export DEEPSEEK_API_KEY="sua-chave-deepseek"'
echo ""
echo "Depois execute: source ~/.bashrc"
