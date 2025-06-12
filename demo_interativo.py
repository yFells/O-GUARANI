"""
Demo Interativo do Chatbot "O Guarani"
Execução simplificada para teste imediato
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChatbotGuaraniDemo:
    """Versão simplificada do chatbot para demonstração"""
    
    def __init__(self):
        print("🤖 Inicializando Chatbot O Guarani (Demo)")
        
        # Base de conhecimento simplificada
        self.base_conhecimento = {
            "peri": "Peri é o protagonista de 'O Guarani', um índio goitacá de força excepcional e lealdade inquebrantável. Ele é devotado a Cecília e representa o 'bom selvagem' idealizado por José de Alencar.",
            
            "cecilia": "Cecília, conhecida como Ceci, é a filha de Dom Antônio de Mariz. É uma jovem bela e bondosa que desenvolve sentimentos especiais por Peri. Representa a pureza e inocência no romance.",
            
            "dom_antonio": "Dom Antônio de Mariz é um nobre português que se estabeleceu no Brasil. Possui um castelo às margens do rio Paquequer e é pai de Cecília. É um homem honrado que respeita Peri.",
            
            "enredo": "O romance se passa no século XVII durante a colonização. Conta a história de amor entre Peri e Cecília, os conflitos com os aimorés e culmina com a destruição do castelo e fuga dos protagonistas.",
            
            "alvaro": "Álvaro é um jovem português, primo de Cecília. Representa o europeu civilizado e desenvolve respeito mútuo por Peri. É corajoso e leal aos habitantes do castelo.",
            
            "aimores": "Os aimorés são uma tribo guerreira e feroz que representa ameaça constante aos habitantes do castelo. São os antagonistas principais da história.",
            
            "natureza": "A natureza brasileira é descrita detalhadamente por Alencar, incluindo florestas tropicais, rios e fauna diversificada, evidenciando sua visão romântica da paisagem nacional.",
            
            "temas": "A obra explora temas como amor impossível, lealdade, sacrifício e choque entre civilizações. Retrata o índio como 'bom selvagem' e idealiza sua pureza moral."
        }
        
        # Preparar sistema de busca
        self._preparar_busca()
        
        # Histórico de conversas
        self.historico = []
        
        print("✅ Sistema pronto para consultas!")
    
    def _preparar_busca(self):
        """Prepara o sistema de busca com TF-IDF"""
        # Criar lista de documentos
        documentos = list(self.base_conhecimento.values())
        chaves = list(self.base_conhecimento.keys())
        
        # Configurar vectorizador
        self.vectorizador = TfidfVectorizer(
            lowercase=True,
            stop_words=['de', 'da', 'do', 'das', 'dos', 'a', 'o', 'as', 'os', 'e', 'ou', 'mas', 'que', 'para', 'com', 'por', 'em', 'no', 'na', 'nos', 'nas'],
            ngram_range=(1, 2)
        )
        
        # Criar matriz de vetores
        self.matriz_docs = self.vectorizador.fit_transform(documentos)
        self.chaves = chaves
    
    def _preprocessar_pergunta(self, pergunta):
        """Pré-processa a pergunta do usuário"""
        # Converter para minúsculas
        pergunta = pergunta.lower()
        
        # Remover pontuação
        pergunta = re.sub(r'[^\w\s]', ' ', pergunta)
        
        # Remover espaços extras
        pergunta = ' '.join(pergunta.split())
        
        return pergunta
    
    def buscar_resposta(self, pergunta):
        """Busca a melhor resposta para a pergunta"""
        # Pré-processar pergunta
        pergunta_processada = self._preprocessar_pergunta(pergunta)
        
        # Vetorizar pergunta
        vetor_pergunta = self.vectorizador.transform([pergunta_processada])
        
        # Calcular similaridades
        similaridades = cosine_similarity(vetor_pergunta, self.matriz_docs).flatten()
        
        # Encontrar melhor match
        melhor_indice = np.argmax(similaridades)
        melhor_similaridade = similaridades[melhor_indice]
        
        # Definir resposta
        if melhor_similaridade > 0.1:  # Limiar de confiança
            chave = self.chaves[melhor_indice]
            resposta = self.base_conhecimento[chave]
            confianca = "Alta" if melhor_similaridade > 0.5 else "Moderada" if melhor_similaridade > 0.3 else "Baixa"
        else:
            resposta = "Desculpe, não encontrei informações específicas sobre sua pergunta em minha base sobre 'O Guarani'. Tente reformular a pergunta."
            confianca = "N/A"
        
        # Registrar no histórico
        registro = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'pergunta': pergunta,
            'resposta': resposta,
            'similaridade': melhor_similaridade,
            'confianca': confianca
        }
        self.historico.append(registro)
        
        return resposta, melhor_similaridade, confianca
    
    def chat_interativo(self):
        """Interface de chat interativo"""
        print("\n" + "="*60)
        print("💬 CHAT INTERATIVO - O GUARANI")
        print("="*60)
        print("Faça perguntas sobre a obra 'O Guarani' de José de Alencar")
        print("Digite 'sair' para encerrar ou 'historico' para ver conversas anteriores")
        print("="*60)
        
        while True:
            try:
                pergunta = input("\n🙋 Você: ").strip()
                
                if not pergunta:
                    continue
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Obrigado por usar o Chatbot O Guarani!")
                    break
                
                if pergunta.lower() == 'historico':
                    self.mostrar_historico()
                    continue
                
                # Buscar resposta
                resposta, similaridade, confianca = self.buscar_resposta(pergunta)
                
                # Exibir resposta
                print(f"\n🤖 Chatbot: {resposta}")
                print(f"   📊 Confiança: {confianca} (similaridade: {similaridade:.3f})")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando chat...")
                break
    
    def mostrar_historico(self):
        """Mostra o histórico de conversas"""
        if not self.historico:
            print("📝 Nenhuma conversa no histórico ainda.")
            return
        
        print("\n" + "="*50)
        print("📚 HISTÓRICO DE CONVERSAS")
        print("="*50)
        
        for i, conv in enumerate(self.historico, 1):
            print(f"\n{i}. [{conv['timestamp']}] {conv['pergunta']}")
            print(f"   🤖 {conv['resposta'][:100]}...")
            print(f"   📊 Confiança: {conv['confianca']} ({conv['similaridade']:.3f})")
    
    def teste_automatico(self):
        """Executa teste automático com perguntas predefinidas"""
        perguntas_teste = [
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Qual o enredo do livro?",
            "Quem é Dom Antônio de Mariz?",
            "O que são os aimorés?",
            "Como é descrita a natureza?",
            "Quais os principais temas da obra?"
        ]
        
        print("\n" + "="*50)
        print("🧪 TESTE AUTOMÁTICO")
        print("="*50)
        
        for pergunta in perguntas_teste:
            resposta, similaridade, confianca = self.buscar_resposta(pergunta)
            print(f"\n❓ {pergunta}")
            print(f"🤖 {resposta[:100]}...")
            print(f"📊 {confianca} ({similaridade:.3f})")
    
    def demonstrar_funcionalidades(self):
        """Demonstra todas as funcionalidades do chatbot"""
        print("🎯 DEMONSTRAÇÃO COMPLETA DO CHATBOT O GUARANI")
        print("="*60)
        
        # 1. Teste automático
        self.teste_automatico()
        
        # 2. Mostrar estatísticas
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"   Base de conhecimento: {len(self.base_conhecimento)} tópicos")
        print(f"   Vocabulário: {len(self.vectorizador.vocabulary_)} termos")
        print(f"   Conversas realizadas: {len(self.historico)}")
        
        # 3. Oferecer chat interativo
        print(f"\n🎮 Deseja iniciar o chat interativo? (s/n)")
        resposta = input().strip().lower()
        if resposta in ['s', 'sim', 'y', 'yes']:
            self.chat_interativo()


def main():
    """Função principal para demonstração"""
    print("🚀 INICIANDO DEMO DO CHATBOT O GUARANI")
    print("="*50)
    
    # Criar instância do chatbot
    chatbot = ChatbotGuaraniDemo()
    
    # Menu de opções
    while True:
        print("\n🎯 OPÇÕES DISPONÍVEIS:")
        print("1. Teste automático")
        print("2. Chat interativo")
        print("3. Demonstração completa")
        print("4. Ver histórico")
        print("5. Sair")
        
        opcao = input("\nEscolha uma opção (1-5): ").strip()
        
        if opcao == '1':
            chatbot.teste_automatico()
        elif opcao == '2':
            chatbot.chat_interativo()
        elif opcao == '3':
            chatbot.demonstrar_funcionalidades()
        elif opcao == '4':
            chatbot.mostrar_historico()
        elif opcao == '5':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")

# Execução rápida para teste
if __name__ == "__main__":
    # Execução direta com teste automático
    chatbot = ChatbotGuaraniDemo()
    print("\n🎯 EXECUTANDO TESTE RÁPIDO...")
    chatbot.teste_automatico()
    
    print("\n" + "="*50)
    print("✅ Demo executado com sucesso!")
    print("Para chat interativo, execute: chatbot.chat_interativo()")
    print("Para menu completo, execute: main()")
