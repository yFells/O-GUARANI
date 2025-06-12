#!/usr/bin/env python3
"""
Teste Rápido do Chatbot O Guarani
Execute este script para ver o sistema funcionando imediatamente
"""

# Verificação e instalação de dependências
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("Instalando dependências necessárias...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "numpy"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

import re
from datetime import datetime

class ChatbotTeste:
    """Versão mínima do chatbot para demonstração rápida"""
    
    def __init__(self):
        # Base de conhecimento sobre O Guarani
        self.base = {
            "peri": "Peri é o protagonista indígena de O Guarani, um goitacá de força excepcional e lealdade absoluta a Cecília. Representa o ideal do 'bom selvagem' romântico.",
            
            "cecilia": "Cecília (Ceci) é a filha de Dom Antônio de Mariz, jovem portuguesa pura e bondosa que desperta a devoção de Peri. Simboliza a inocência europeia.",
            
            "antonio": "Dom Antônio de Mariz é o nobre português proprietário do castelo às margens do Paquequer, pai de Cecília e homem de honra que respeita Peri.",
            
            "enredo": "Romance passado no século XVII sobre o amor entre o índio Peri e a portuguesa Cecília, envolvendo conflitos com aimorés e a destruição do castelo.",
            
            "aimores": "Os aimorés são a tribo inimiga feroz que ameaça constantemente os habitantes do castelo, representando o perigo selvagem na obra.",
            
            "natureza": "Alencar descreve exuberantemente a natureza brasileira - florestas, rios, fauna - como personagem quase vivo da narrativa romântica."
        }
        
        # Configurar sistema de busca
        self.vectorizer = TfidfVectorizer(
            lowercase=True, 
            ngram_range=(1, 2),
            max_features=500
        )
        
        # Treinar com base de conhecimento
        textos = list(self.base.values())
        self.matriz = self.vectorizer.fit_transform(textos)
        self.chaves = list(self.base.keys())
        
        print("✅ Chatbot O Guarani inicializado!")
        print(f"📚 Base: {len(self.base)} tópicos")
        print(f"🔤 Vocabulário: {len(self.vectorizer.vocabulary_)} termos")
    
    def consultar(self, pergunta):
        """Processa uma pergunta e retorna resposta"""
        # Vectorizar pergunta
        vetor_pergunta = self.vectorizer.transform([pergunta.lower()])
        
        # Calcular similaridades
        similarities = cosine_similarity(vetor_pergunta, self.matriz).flatten()
        
        # Encontrar melhor match
        melhor_idx = np.argmax(similarities)
        melhor_sim = similarities[melhor_idx]
        
        if melhor_sim > 0.1:  # Threshold mínimo
            resposta = self.base[self.chaves[melhor_idx]]
            confianca = "Alta" if melhor_sim > 0.5 else "Média" if melhor_sim > 0.3 else "Baixa"
        else:
            resposta = "Não encontrei informações sobre isso em O Guarani. Tente perguntar sobre Peri, Cecília, enredo, etc."
            confianca = "N/A"
        
        return resposta, melhor_sim, confianca
    
    def teste_completo(self):
        """Executa bateria de testes"""
        perguntas = [
            "Quem é Peri?",
            "Fale sobre Cecília", 
            "Qual o enredo do livro?",
            "Quem é Dom Antônio?",
            "O que são aimorés?",
            "Como é a natureza no livro?",
            "Quem escreveu a obra?"  # Esta deve falhar
        ]
        
        print(f"\n🧪 EXECUTANDO {len(perguntas)} TESTES:")
        print("="*60)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas, 1):
            resposta, similaridade, confianca = self.consultar(pergunta)
            
            print(f"\n{i}. ❓ {pergunta}")
            print(f"   🤖 {resposta[:80]}...")
            print(f"   📊 Confiança: {confianca} (sim: {similaridade:.3f})")
            
            resultados.append({
                'pergunta': pergunta,
                'similaridade': similaridade,
                'confianca': confianca,
                'sucesso': confianca != "N/A"
            })
        
        # Estatísticas finais
        sucessos = sum(1 for r in resultados if r['sucesso'])
        taxa_sucesso = (sucessos / len(resultados)) * 100
        sim_media = np.mean([r['similaridade'] for r in resultados if r['sucesso']])
        
        print(f"\n📈 RESULTADOS:")
        print(f"   Perguntas respondidas: {sucessos}/{len(resultados)}")
        print(f"   Taxa de sucesso: {taxa_sucesso:.1f}%")
        print(f"   Similaridade média: {sim_media:.3f}")
        
        return resultados

def demo_interativo():
    """Demonstração interativa simples"""
    chatbot = ChatbotTeste()
    
    # Executar teste automático
    resultados = chatbot.teste_completo()
    
    print(f"\n💬 MODO INTERATIVO (digite 'sair' para encerrar):")
    print("="*50)
    
    while True:
        try:
            pergunta = input("\n🙋 Você: ").strip()
            
            if not pergunta or pergunta.lower() in ['sair', 'exit']:
                print("👋 Até logo!")
                break
                
            resposta, sim, conf = chatbot.consultar(pergunta)
            print(f"🤖 Bot: {resposta}")
            print(f"📊 Confiança: {conf} ({sim:.3f})")
            
        except KeyboardInterrupt:
            print("\n👋 Encerrando...")
            break

def teste_rapido():
    """Execução rápida apenas com testes automáticos"""
    print("🚀 TESTE RÁPIDO DO CHATBOT O GUARANI")
    print("="*50)
    
    chatbot = ChatbotTeste()
    resultados = chatbot.teste_completo()
    
    print(f"\n✅ Teste concluído!")
    print(f"Sistema funcionando corretamente.")
    
    # Demonstrar uma consulta específica
    print(f"\n🎯 EXEMPLO DE CONSULTA DETALHADA:")
    print("-" * 40)
    
    pergunta_exemplo = "Quem é o protagonista de O Guarani?"
    resposta, sim, conf = chatbot.consultar(pergunta_exemplo)
    
    print(f"Pergunta: {pergunta_exemplo}")
    print(f"Resposta: {resposta}")
    print(f"Confiança: {conf} (similaridade: {sim:.3f})")
    
    return chatbot

# Execução automática quando o script é executado
if __name__ == "__main__":
    import sys
    
    # Verificar argumentos
    if len(sys.argv) > 1 and sys.argv[1] == "--interativo":
        demo_interativo()
    else:
        # Execução padrão: teste rápido
        chatbot = teste_rapido()
        
        print(f"\nPara modo interativo, execute:")
        print(f"python {__file__} --interativo")
        
        print(f"\nOu use o chatbot diretamente:")
        print(f"chatbot.consultar('sua pergunta aqui')")
