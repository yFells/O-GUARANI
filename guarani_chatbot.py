"""
Chatbot "O Guarani" - Sistema de PLN para consultas sobre a obra de José de Alencar
Implementação completa das 5 fases descritas no projeto
"""

import os
import numpy as np
import pandas as pd
import re
import requests
from typing import List, Dict, Tuple
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Bibliotecas de PLN
try:
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import RSLPStemmer
except ImportError:
    print("Instalando bibliotecas necessárias...")
    os.system("pip install spacy nltk scikit-learn matplotlib seaborn")
    import spacy
    import nltk

# Downloads necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/rslp')
except LookupError:
    print("Baixando recursos do NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')

class GuaraniChatbot:
    """
    Chatbot especializado em responder perguntas sobre "O Guarani" de José de Alencar
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani...")
        
        # Configurações
        self.chunk_size = 250  # palavras por chunk
        self.overlap = 0.5     # 50% de sobreposição
        self.similarity_threshold = 0.3  # limiar mínimo de similaridade
        self.top_chunks = 3    # top chunks para resposta
        
        # Componentes do sistema
        self.stemmer = RSLPStemmer()
        self.stop_words = set(stopwords.words('portuguese'))
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2)
        )
        
        # Dados do sistema
        self.text_chunks = []
        self.chunk_vectors = None
        self.original_text = ""
        
        # Histórico
        self.conversation_history = []
        self.processing_log = []
        
        self._log("Sistema inicializado com sucesso")
    
    def _log(self, message: str):
        """Registra eventos no histórico de processamento"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(f"📝 {log_entry}")
    
    def fase1_preparar_ambiente(self):
        """Fase 1: Preparação do ambiente e obtenção dos dados"""
        self._log("=== FASE 1: PREPARAÇÃO DO AMBIENTE ===")
        
        # Simulando obtenção do texto (normalmente seria um arquivo)
        sample_text = """
        O Guarani é um romance de José de Alencar publicado em 1857. A história se passa no século XVII, 
        durante a colonização do Brasil. O protagonista é Peri, um índio goitacá de força excepcional 
        e lealdade inquebrantável.
        
        Peri é caracterizado por sua devoção a Cecília (Ceci), filha do fidalgo português Dom Antônio de Mariz. 
        A relação entre Peri e Ceci representa o encontro entre duas culturas: a indígena e a europeia.
        
        Dom Antônio de Mariz é um nobre português que se estabeleceu no Brasil com sua família. 
        Ele possui um castelo fortificado às margens do rio Paquequer, onde vive com sua esposa, 
        filhos e alguns agregados.
        
        A obra retrata os conflitos entre diferentes grupos: os portugueses colonizadores, 
        os índios aimorés (inimigos de Peri) e os aventureiros que buscam ouro na região.
        
        Álvaro é um jovem português, primo de Cecília, que também habita o castelo. 
        Ele representa o europeu civilizado, contrastando com a natureza selvagem de Peri.
        
        Isabel é irmã de Cecília, uma jovem impetuosa que se apaixona por Álvaro. 
        Sua história pessoal adiciona complexidade aos relacionamentos na narrativa.
        
        Os aimorés são os antagonistas principais da história, representando o perigo 
        constante que ameaça a segurança dos habitantes do castelo.
        
        A natureza brasileira é quase um personagem na obra, sendo descrita com detalhes 
        que evidenciam a visão romântica de Alencar sobre a paisagem nacional.
        
        O romance explora temas como o amor impossível entre Peri e Ceci, 
        a lealdade, o sacrifício e o choque entre civilizações.
        
        A linguagem de Alencar mescla o português culto com expressões que buscam 
        retratar a fala dos personagens indígenas, criando um estilo único.
        """
        
        self.original_text = sample_text
        self._log(f"Texto carregado: {len(sample_text)} caracteres")
        return True
    
    def fase2_processar_dados(self):
        """Fase 2: Processamento e estruturação dos dados"""
        self._log("=== FASE 2: PROCESSAMENTO DOS DADOS ===")
        
        # Limpeza do texto
        cleaned_text = self._clean_text(self.original_text)
        self._log("Texto limpo e normalizado")
        
        # Tokenização e segmentação em chunks
        self.text_chunks = self._create_chunks(cleaned_text)
        self._log(f"Criados {len(self.text_chunks)} chunks de texto")
        
        # Processamento dos chunks
        processed_chunks = []
        for i, chunk in enumerate(self.text_chunks):
            processed = self._preprocess_text(chunk)
            processed_chunks.append(processed)
        
        self._log("Chunks pré-processados (tokenização, remoção de stopwords)")
        return processed_chunks
    
    def fase3_armazenar_indexar(self, processed_chunks: List[str]):
        """Fase 3: Armazenamento e indexação"""
        self._log("=== FASE 3: ARMAZENAMENTO E INDEXAÇÃO ===")
        
        # Criação de vetores TF-IDF (substituto simplificado do Word2Vec)
        self.chunk_vectors = self.vectorizer.fit_transform(processed_chunks)
        self._log(f"Vetores criados: {self.chunk_vectors.shape}")
        
        # Salvamento dos dados
        self._save_data()
        self._log("Dados indexados e salvos")
        
        return True
    
    def fase4_mecanismo_resposta(self, pergunta: str) -> str:
        """Fase 4: Mecanismo de recuperação e geração de resposta"""
        self._log(f"=== CONSULTANDO: {pergunta} ===")
        
        # Pré-processamento da pergunta
        processed_question = self._preprocess_text(pergunta)
        question_vector = self.vectorizer.transform([processed_question])
        
        # Busca por similaridade
        similarities = cosine_similarity(question_vector, self.chunk_vectors).flatten()
        
        # Ranqueamento e filtragem
        relevant_chunks = []
        for i, similarity in enumerate(similarities):
            if similarity >= self.similarity_threshold:
                relevant_chunks.append({
                    'chunk_id': i,
                    'text': self.text_chunks[i],
                    'similarity': similarity
                })
        
        # Ordenação por similaridade
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = relevant_chunks[:self.top_chunks]
        
        self._log(f"Encontrados {len(relevant_chunks)} chunks relevantes")
        
        # Geração da resposta
        if not top_chunks:
            response = "Desculpe, não encontrei informações relevantes sobre sua pergunta."
        else:
            response = self._generate_response(pergunta, top_chunks)
        
        # Registro no histórico
        self.conversation_history.append({
            'pergunta': pergunta,
            'resposta': response,
            'chunks_usados': len(top_chunks),
            'similaridade_max': max([c['similarity'] for c in top_chunks]) if top_chunks else 0
        })
        
        return response
    
    def fase5_interface_usuario(self):
        """Fase 5: Interface com o usuário"""
        self._log("=== FASE 5: INTERFACE ATIVA ===")
        
        print("\n" + "="*60)
        print("🤖 CHATBOT O GUARANI")
        print("Especialista na obra de José de Alencar")
        print("Digite 'sair' para encerrar")
        print("="*60)
        
        while True:
            try:
                pergunta = input("\n💬 Sua pergunta: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                
                if not pergunta:
                    continue
                
                resposta = self.fase4_mecanismo_resposta(pergunta)
                print(f"\n🤖 Resposta: {resposta}")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto"""
        # Remove quebras de linha desnecessárias
        text = re.sub(r'\n+', ' ', text)
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        # Remove caracteres especiais mantendo pontuação básica
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.strip()
    
    def _preprocess_text(self, text: str) -> str:
        """Pré-processa texto: tokenização, remoção de stopwords, stemming"""
        # Tokenização
        tokens = word_tokenize(text.lower(), language='portuguese')
        
        # Remoção de stopwords e tokens muito pequenos
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2 and token.isalpha()
        ]
        
        # Stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
        
        return ' '.join(stemmed_tokens)
    
    def _create_chunks(self, text: str) -> List[str]:
        """Cria chunks de texto com sobreposição"""
        sentences = sent_tokenize(text, language='portuguese')
        chunks = []
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            
            if current_word_count + len(words) <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += len(words)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Sobreposição: mantém parte do chunk anterior
                overlap_sentences = int(len(current_chunk) * self.overlap)
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk.append(sentence)
                current_word_count = sum(len(s.split()) for s in current_chunk)
        
        # Adiciona o último chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_response(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta baseada nos chunks mais relevantes"""
        # Template de resposta
        intro = "Com base na obra 'O Guarani':\n\n"
        
        # Extrai informações dos chunks mais relevantes
        main_content = chunks[0]['text']
        
        # Formatação simples da resposta
        response = intro + main_content
        
        # Adiciona informação de confiança
        confidence = chunks[0]['similarity']
        if confidence > 0.7:
            confidence_text = " (Alta confiança)"
        elif confidence > 0.5:
            confidence_text = " (Confiança moderada)"
        else:
            confidence_text = " (Baixa confiança)"
        
        return response + confidence_text
    
    def _save_data(self):
        """Salva os dados processados"""
        data = {
            'chunks': self.text_chunks,
            'vectors': self.chunk_vectors,
            'vectorizer': self.vectorizer
        }
        
        # Em um sistema real, salvaria em arquivo
        # pickle.dump(data, open('guarani_data.pkl', 'wb'))
        self._log("Dados salvos em memória")
    
    def executar_sistema_completo(self):
        """Executa todas as fases do sistema"""
        try:
            # Fase 1: Preparação
            if not self.fase1_preparar_ambiente():
                raise Exception("Erro na Fase 1")
            
            # Fase 2: Processamento
            processed_chunks = self.fase2_processar_dados()
            if not processed_chunks:
                raise Exception("Erro na Fase 2")
            
            # Fase 3: Indexação
            if not self.fase3_armazenar_indexar(processed_chunks):
                raise Exception("Erro na Fase 3")
            
            # Fase 4 e 5: Sistema ativo
            self._log("Sistema pronto para consultas!")
            
            # Exemplos de teste
            self.testar_sistema()
            
            return True
            
        except Exception as e:
            self._log(f"Erro na execução: {e}")
            return False
    
    def testar_sistema(self):
        """Testa o sistema com perguntas exemplo"""
        perguntas_teste = [
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Qual é o enredo do livro?",
            "Quem são os personagens principais?",
            "Onde se passa a história?"
        ]
        
        print("\n" + "="*50)
        print("🧪 TESTE AUTOMÁTICO DO SISTEMA")
        print("="*50)
        
        for pergunta in perguntas_teste:
            print(f"\n❓ Pergunta: {pergunta}")
            resposta = self.fase4_mecanismo_resposta(pergunta)
            print(f"🤖 Resposta: {resposta[:200]}...")
    
    def mostrar_historico(self):
        """Mostra o histórico de processamento e conversas"""
        print("\n" + "="*50)
        print("📊 HISTÓRICO DE PROCESSAMENTO")
        print("="*50)
        
        for log in self.processing_log:
            print(log)
        
        if self.conversation_history:
            print("\n" + "="*50)
            print("💬 HISTÓRICO DE CONVERSAS")
            print("="*50)
            
            for i, conv in enumerate(self.conversation_history, 1):
                print(f"\n{i}. {conv['pergunta']}")
                print(f"   Resposta: {conv['resposta'][:100]}...")
                print(f"   Chunks usados: {conv['chunks_usados']}")
                print(f"   Similaridade: {conv['similaridade_max']:.3f}")
    
    def estatisticas_sistema(self):
        """Mostra estatísticas do sistema"""
        print("\n" + "="*50)
        print("📈 ESTATÍSTICAS DO SISTEMA")
        print("="*50)
        
        print(f"Total de chunks: {len(self.text_chunks)}")
        print(f"Tamanho do vocabulário: {len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 'N/A'}")
        print(f"Consultas realizadas: {len(self.conversation_history)}")
        print(f"Threshold de similaridade: {self.similarity_threshold}")
        print(f"Eventos de log: {len(self.processing_log)}")


# Função principal para execução
def main():
    """Função principal para executar o chatbot"""
    chatbot = GuaraniChatbot()
    
    print("Executando sistema completo...")
    if chatbot.executar_sistema_completo():
        print("\n✅ Sistema executado com sucesso!")
        
        # Mostra estatísticas
        chatbot.estatisticas_sistema()
        
        # Mostra histórico
        chatbot.mostrar_historico()
        
        # Oferece interface interativa
        print("\nDeseja iniciar a interface de chat? (s/n)")
        resposta = input().strip().lower()
        if resposta in ['s', 'sim', 'y', 'yes']:
            chatbot.fase5_interface_usuario()
    
    else:
        print("❌ Erro na execução do sistema")

# Execução
if __name__ == "__main__":
    main()
