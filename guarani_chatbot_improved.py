"""
Chatbot "O Guarani" - Sistema de PLN para consultas sobre a obra de José de Alencar
Versão melhorada com embeddings semânticos e otimizações
"""

import os
import numpy as np
import pandas as pd
import re
import pickle
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Bibliotecas de PLN
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import RSLPStemmer
except ImportError:
    print("⚠️ NLTK não encontrado. Usando processamento simplificado.")
    nltk = None

# Sentence Transformers para embeddings semânticos
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("⚠️ Sentence-transformers não encontrado. Usando TF-IDF como fallback.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    EMBEDDINGS_AVAILABLE = False

class GuaraniChatbotImproved:
    """
    Chatbot especializado em responder perguntas sobre "O Guarani" de José de Alencar
    Versão melhorada com embeddings semânticos
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani (Versão Melhorada)...")
        
        # Configurações otimizadas
        self.chunk_size = 150      # Reduzido para chunks mais focados
        self.overlap = 0.3         # Sobreposição reduzida mas suficiente
        self.similarity_threshold = 0.15  # Limiar mais restritivo
        self.top_chunks = 3        # Top chunks para resposta
        self.sentence_level = True # Nova feature: busca no nível de sentença
        
        # Inicializar estruturas de dados primeiro
        self.conversation_history = []
        self.processing_log = []
        self.performance_metrics = []
        
        # Dados do sistema
        self.text_chunks = []
        self.chunk_vectors = None
        self.original_text = ""
        self.sentences_per_chunk = []  # Novo: mapeamento de sentenças por chunk
        
        # Inicialização de componentes PLN (após inicializar processing_log)
        self._init_nlp_components()
        
        self._log("Sistema inicializado com sucesso")
    
    def _init_nlp_components(self):
        """Inicializa componentes de PLN com fallbacks"""
        # Stop words
        try:
            if nltk:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('rslp', quiet=True)
                self.stop_words = set(stopwords.words('portuguese'))
                self.stemmer = RSLPStemmer()
                self._log("NLTK inicializado com sucesso")
            else:
                raise ImportError("NLTK não disponível")
        except Exception as e:
            # Fallback: stop words básicas em português
            self.stop_words = {
                'a', 'o', 'e', 'de', 'da', 'do', 'em', 'um', 'uma', 'com', 'para',
                'por', 'que', 'se', 'na', 'no', 'ao', 'aos', 'as', 'os', 'mais',
                'mas', 'ou', 'ter', 'ser', 'estar', 'seu', 'sua', 'seus', 'suas'
            }
            self.stemmer = None
            self._log(f"Usando stop words básicas (NLTK indisponível: {e})")
        
        # Modelo de embeddings ou TF-IDF
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_embeddings = True
                self._log("Usando embeddings semânticos (SentenceTransformers)")
            except Exception as e:
                self._log(f"Erro ao carregar SentenceTransformers: {e}")
                self._init_tfidf_fallback()
        else:
            self._log("SentenceTransformers não disponível, usando TF-IDF")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Inicializa TF-IDF como fallback"""
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words=list(self.stop_words),  # Melhoramento: reintroduzir stop words
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        self.use_embeddings = False
        self._log("Usando TF-IDF como método de vetorização")
    
    def _log(self, message: str):
        """Registra eventos no histórico de processamento"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Verificação de segurança para evitar erros de inicialização
        if hasattr(self, 'processing_log'):
            self.processing_log.append(log_entry)
        else:
            # Fallback se processing_log ainda não foi inicializado
            print(f"⚠️  Log antes da inicialização: {log_entry}")
        
        print(f"📝 {log_entry}")
    
    def fase1_preparar_ambiente(self):
        """Fase 1: Preparação do ambiente e obtenção dos dados"""
        self._log("=== FASE 1: PREPARAÇÃO DO AMBIENTE ===")
        
        # Carregamento do texto
        texto_carregado = self._carregar_texto_guarani()
        
        if texto_carregado:
            self.original_text = texto_carregado
            self._log(f"Texto carregado: {len(texto_carregado)} caracteres")
            
            # Estatísticas detalhadas
            stats = self._analyze_text_stats(texto_carregado)
            for key, value in stats.items():
                self._log(f"{key}: {value}")
            
            return True
        else:
            self._log("Erro: Não foi possível carregar o texto")
            return False
    
    def _analyze_text_stats(self, text: str) -> Dict:
        """Analisa estatísticas detalhadas do texto"""
        palavras = text.split()
        linhas = text.splitlines()
        
        # Análise de sentenças
        try:
            if nltk:
                sentences = sent_tokenize(text, language='portuguese')
            else:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
        except:
            sentences = text.split('.')
        
        return {
            "Palavras": len(palavras),
            "Linhas": len(linhas),
            "Sentenças": len(sentences),
            "Caracteres": len(text),
            "Palavras únicas": len(set([p.lower() for p in palavras if p.isalpha()])),
            "Média palavras/sentença": round(len(palavras) / len(sentences), 2) if sentences else 0
        }
    
    def _carregar_texto_guarani(self):
        """Carrega o texto com melhor tratamento de encoding"""
        # Primeiro, tenta carregar de arquivo
        for filename in ['guarani.txt', 'o_guarani.txt', 'alencar_guarani.txt']:
            result = self._try_load_file(filename)
            if result:
                return result
        
        # Se não encontrou arquivo, usa texto de demonstração expandido
        self._log("Arquivo não encontrado. Usando texto de demonstração expandido.")
        return self._texto_demo_expandido()
    
    def _try_load_file(self, filename: str):
        """Tenta carregar arquivo com diferentes encodings"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as arquivo:
                    texto = arquivo.read()
                    if len(texto) > 1000:  # Validação de tamanho mínimo
                        self._log(f"Arquivo '{filename}' carregado com encoding: {encoding}")
                        return texto
                    else:
                        self._log(f"Arquivo '{filename}' muito pequeno ({len(texto)} chars)")
            except FileNotFoundError:
                continue
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._log(f"Erro ao ler '{filename}': {e}")
                continue
        
        return None
    
    def _texto_demo_expandido(self):
        """Texto de demonstração mais rico e detalhado"""
        return """
        O Guarani é um romance indianista de José de Alencar, publicado em 1857. A narrativa se desenvolve no século XVII, 
        durante o período colonial brasileiro, nas montanhas fluminenses próximas ao rio Paquequer.
        
        Peri é o protagonista da obra, um índio goitacá de força hercúlea e lealdade inabalável. Ele é descrito como um 
        guerreiro corajoso, de estatura imponente e caráter nobre. Peri demonstra uma devoção absoluta a Cecília (Ceci), 
        filha do fidalgo português Dom Antônio de Mariz. Esta devoção representa o amor impossível entre duas raças distintas.
        
        Cecília, chamada carinhosamente de Ceci, é uma jovem portuguesa de beleza singular e caráter doce. Ela é filha 
        de Dom Antônio de Mariz e representa a pureza e a inocência feminina idealizadas pelo Romantismo. Ceci desenvolve 
        sentimentos fraternais por Peri, vendo nele um protetor dedicado.
        
        Dom Antônio de Mariz é um nobre português, fidalgo da Casa Real, que se estabeleceu no Brasil após cometer um crime 
        de honra em Portugal. Ele construiu um castelo fortificado nas margens do rio Paquequer, onde vive com sua família. 
        Dom Antônio é caracterizado como um homem honrado, mas marcado pelo passado.
        
        Dona Lauriana é a esposa de Dom Antônio, uma senhora portuguesa de origem nobre. Ela representa os valores 
        aristocráticos europeus e inicialmente demonstra preconceito em relação aos indígenas.
        
        Álvaro é um jovem português, primo de Cecília, que também habita o castelo. Ele encarna o ideal do cavaleiro 
        medieval, sendo corajoso, nobre e apaixonado por Ceci. Álvaro representa a civilização europeia em contraste 
        com a natureza selvagem de Peri.
        
        Isabel é irmã de Cecília, uma jovem impetuosa e apaixonada. Ela se enamora de Álvaro, criando um triângulo 
        amoroso que adiciona complexidade às relações familiares. Isabel possui um temperamento mais forte que sua irmã.
        
        Loredano é um dos antagonistas da história, um aventureiro italiano que se infiltra no castelo com intenções 
        malévolas. Ele planeja assassinar Dom Antônio e se apossar de suas riquezas, representando a traição e a vilania.
        
        Os aimorés são a tribo indígena antagonista, inimigos mortais de Peri e de sua tribo goitacá. Eles representam 
        o perigo constante que ameaça a segurança dos habitantes do castelo. Os aimorés são descritos como selvagens 
        e canibais, contrastando com a nobreza de Peri.
        
        A natureza brasileira desempenha papel fundamental na narrativa, sendo descrita com exuberância e riqueza de 
        detalhes. Alencar retrata as florestas, rios e montanhas como cenário épico que reflete o caráter dos personagens. 
        A paisagem tropical serve como pano de fundo para os conflitos entre civilização e barbárie.
        
        O romance explora temas centrais como o amor impossível entre raças diferentes, representado pela relação entre 
        Peri e Ceci. A lealdade e o sacrifício são exemplificados pela devoção absoluta do índio à família Mariz. 
        O choque entre civilizações aparece no contraste entre os valores europeus e indígenas.
        
        A linguagem de Alencar combina o português erudito com tentativas de recriar a fala indígena, criando um estilo 
        único que busca expressar a realidade brasileira. O autor emprega descrições românticas e idealizadas tanto 
        dos personagens quanto da natureza.
        
        O desfecho trágico da obra culmina com a destruição do castelo e a fuga de Peri e Ceci, simbolizando o nascimento 
        de uma nova raça brasileira através da união simbólica entre o índio e a portuguesa. Esta união representa a 
        formação da identidade nacional brasileira segundo a visão romântica de Alencar.
        
        O Guarani tornou-se uma das obras mais importantes do Romantismo brasileiro, influenciando a literatura nacional 
        e contribuindo para a construção do mito do "bom selvagem" e da identidade cultural brasileira.
        """
    
    def fase2_processar_dados(self):
        """Fase 2: Processamento otimizado dos dados"""
        self._log("=== FASE 2: PROCESSAMENTO AVANÇADO DOS DADOS ===")
        
        # Limpeza aprimorada
        cleaned_text = self._advanced_text_cleaning(self.original_text)
        self._log("Texto limpo e normalizado")
        
        # Criação de chunks otimizada
        self.text_chunks = self._create_optimized_chunks(cleaned_text)
        self._log(f"Criados {len(self.text_chunks)} chunks otimizados")
        
        # Mapeamento de sentenças por chunk
        self._map_sentences_to_chunks()
        
        return self.text_chunks
    
    def _advanced_text_cleaning(self, text: str) -> str:
        """Limpeza avançada preservando contexto importante"""
        # Normalização de espaços e quebras de linha
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Preservar pontuação importante para segmentação
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        # Remover caracteres especiais mas preservar acentos
        text = re.sub(r'[^\w\sáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ.!?;:,\-]', ' ', text)
        
        # Normalização final
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _create_optimized_chunks(self, text: str) -> List[str]:
        """Criação otimizada de chunks com melhor contexto"""
        # Segmentação em sentenças
        try:
            if nltk:
                sentences = sent_tokenize(text, language='portuguese')
            else:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            # Verificar se a sentença cabe no chunk atual
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                # Finalizar chunk atual se não estiver vazio
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                
                # Começar novo chunk com sobreposição
                overlap_sentences = int(len(current_chunk) * self.overlap)
                if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = []
                    current_word_count = 0
                
                # Adicionar nova sentença
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Adicionar último chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def _map_sentences_to_chunks(self):
        """Mapeia sentenças dentro de cada chunk para busca refinada"""
        self.sentences_per_chunk = []
        
        for chunk in self.text_chunks:
            try:
                if nltk:
                    sentences = sent_tokenize(chunk, language='portuguese')
                else:
                    sentences = re.split(r'[.!?]+', chunk)
                    sentences = [s.strip() for s in sentences if s.strip()]
            except:
                sentences = [s.strip() for s in chunk.split('.') if s.strip()]
            
            self.sentences_per_chunk.append(sentences)
    
    def fase3_armazenar_indexar(self, chunks: List[str]):
        """Fase 3: Indexação com embeddings semânticos ou TF-IDF melhorado"""
        self._log("=== FASE 3: INDEXAÇÃO SEMÂNTICA AVANÇADA ===")
        
        if self.use_embeddings:
            # Usar embeddings semânticos
            self.chunk_vectors = self.embedding_model.encode(
                chunks, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self._log(f"Embeddings criados: {self.chunk_vectors.shape}")
        else:
            # Usar TF-IDF melhorado
            processed_chunks = [self._preprocess_for_tfidf(chunk) for chunk in chunks]
            self.chunk_vectors = self.vectorizer.fit_transform(processed_chunks)
            self._log(f"Vetores TF-IDF criados: {self.chunk_vectors.shape}")
        
        # Salvar dados
        self._save_enhanced_data()
        
        return True
    
    def _preprocess_for_tfidf(self, text: str) -> str:
        """Pré-processamento específico para TF-IDF"""
        text = text.lower()
        
        # Tokenização
        words = re.findall(r'\b\w+\b', text)
        
        # Remoção de stop words e palavras muito curtas
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Stemming se disponível
        if self.stemmer:
            try:
                filtered_words = [self.stemmer.stem(word) for word in filtered_words]
            except:
                pass  # Continuar sem stemming se houver erro
        
        return ' '.join(filtered_words)
    
    def fase4_mecanismo_resposta(self, pergunta: str) -> str:
        """Fase 4: Mecanismo avançado de recuperação e geração"""
        start_time = datetime.now()
        self._log(f"=== CONSULTA: {pergunta} ===")
        
        if self.use_embeddings:
            response = self._resposta_com_embeddings(pergunta)
        else:
            response = self._resposta_com_tfidf(pergunta)
        
        # Métricas de performance
        processing_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.append({
            'pergunta': pergunta,
            'tempo_processamento': processing_time,
            'metodo': 'embeddings' if self.use_embeddings else 'tfidf'
        })
        
        return response
    
    def _resposta_com_embeddings(self, pergunta: str) -> str:
        """Geração de resposta usando embeddings semânticos"""
        # Codificar pergunta
        question_vector = self.embedding_model.encode([pergunta])
        
        # Calcular similaridades
        similarities = cosine_similarity(question_vector, self.chunk_vectors).flatten()
        
        return self._process_similarities(pergunta, similarities)
    
    def _resposta_com_tfidf(self, pergunta: str) -> str:
        """Geração de resposta usando TF-IDF"""
        # Pré-processar pergunta
        processed_question = self._preprocess_for_tfidf(pergunta)
        question_vector = self.vectorizer.transform([processed_question])
        
        # Calcular similaridades
        similarities = cosine_similarity(question_vector, self.chunk_vectors).flatten()
        
        return self._process_similarities(pergunta, similarities)
    
    def _process_similarities(self, pergunta: str, similarities: np.ndarray) -> str:
        """Processa similaridades e gera resposta"""
        max_sim = np.max(similarities) if len(similarities) > 0 else 0
        mean_sim = np.mean(similarities) if len(similarities) > 0 else 0
        
        self._log(f"Similaridade máxima: {max_sim:.3f}, média: {mean_sim:.3f}")
        
        # Encontrar chunks relevantes
        relevant_chunks = []
        for i, similarity in enumerate(similarities):
            if similarity >= self.similarity_threshold:
                relevant_chunks.append({
                    'chunk_id': i,
                    'text': self.text_chunks[i],
                    'similarity': similarity,
                    'sentences': self.sentences_per_chunk[i] if i < len(self.sentences_per_chunk) else []
                })
        
        # Ordenar por similaridade
        relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = relevant_chunks[:self.top_chunks]
        
        self._log(f"Encontrados {len(relevant_chunks)} chunks relevantes")
        
        # Gerar resposta
        if not top_chunks:
            response = self._no_answer_response(pergunta, max_sim)
        else:
            if self.sentence_level and top_chunks:
                response = self._generate_sentence_level_response(pergunta, top_chunks)
            else:
                response = self._generate_chunk_level_response(pergunta, top_chunks)
        
        # Registrar no histórico
        self._update_conversation_history(pergunta, response, top_chunks, max_sim)
        
        return response
    
    def _generate_sentence_level_response(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta no nível de sentença (mais precisa)"""
        # Encontrar a sentença mais relevante dentro dos melhores chunks
        best_sentences = []
        
        for chunk_data in chunks[:2]:  # Usar top 2 chunks
            sentences = chunk_data['sentences']
            if not sentences:
                continue
            
            # Calcular similaridade para cada sentença
            if self.use_embeddings:
                sentence_vectors = self.embedding_model.encode(sentences)
                question_vector = self.embedding_model.encode([pergunta])
                sentence_similarities = cosine_similarity(question_vector, sentence_vectors).flatten()
            else:
                # Para TF-IDF, usar uma aproximação simples
                sentence_similarities = [
                    len(set(pergunta.lower().split()) & set(sent.lower().split())) / 
                    len(set(pergunta.lower().split()) | set(sent.lower().split()))
                    for sent in sentences
                ]
            
            # Encontrar melhor sentença neste chunk
            if sentence_similarities:
                best_idx = np.argmax(sentence_similarities)
                best_sentences.append({
                    'text': sentences[best_idx],
                    'similarity': sentence_similarities[best_idx],
                    'chunk_similarity': chunk_data['similarity']
                })
        
        # Selecionar a melhor sentença geral
        if best_sentences:
            best_sentences.sort(key=lambda x: x['similarity'], reverse=True)
            best_sentence = best_sentences[0]
            
            confidence = self._calculate_confidence(best_sentence['similarity'])
            
            response = f"Com base em 'O Guarani':\n\n{best_sentence['text']}"
            response += f"\n\n{confidence}"
            
            return response
        else:
            # Fallback para resposta por chunk
            return self._generate_chunk_level_response(pergunta, chunks)
    
    def _generate_chunk_level_response(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta no nível de chunk"""
        if len(chunks) == 1:
            main_content = chunks[0]['text']
            intro = "Com base no texto de 'O Guarani':\n\n"
        else:
            # Combinar informações dos melhores chunks
            combined_text = " ".join([chunk['text'] for chunk in chunks[:2]])
            main_content = combined_text
            intro = "Combinando informações de 'O Guarani':\n\n"
        
        # Truncar se muito longo
        if len(main_content) > 600:
            main_content = main_content[:600] + "..."
        
        confidence = self._calculate_confidence(chunks[0]['similarity'])
        
        return intro + main_content + "\n\n" + confidence
    
    def _calculate_confidence(self, similarity: float) -> str:
        """Calcula e retorna indicador de confiança"""
        if similarity > 0.7:
            return "✅ (Confiança muito alta)"
        elif similarity > 0.5:
            return "🟢 (Confiança alta)"
        elif similarity > 0.3:
            return "🟡 (Confiança moderada)"
        elif similarity > 0.15:
            return "🟠 (Confiança baixa - considere reformular)"
        else:
            return "🔴 (Confiança muito baixa)"
    
    def _no_answer_response(self, pergunta: str, max_similarity: float) -> str:
        """Resposta quando não encontra informações relevantes"""
        base_msg = "Não encontrei informações específicas sobre sua pergunta no texto de 'O Guarani'."
        
        if max_similarity > 0.05:
            suggestion = " Tente reformular a pergunta ou ser mais específico sobre personagens, eventos ou temas da obra."
        else:
            suggestion = " Sua pergunta pode estar fora do escopo da obra ou usar termos muito diferentes do texto original."
        
        examples = "\n\nExemplos de perguntas: 'Quem é Peri?', 'Fale sobre Cecília', 'Qual o enredo?', 'Onde se passa a história?'"
        
        return base_msg + suggestion + examples
    
    def _update_conversation_history(self, pergunta: str, resposta: str, chunks: List[Dict], max_sim: float):
        """Atualiza histórico de conversa com mais detalhes"""
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'pergunta': pergunta,
            'resposta': resposta,
            'chunks_usados': len(chunks),
            'similaridade_max': max_sim,
            'metodo': 'embeddings' if self.use_embeddings else 'tfidf',
            'chunks_ids': [c['chunk_id'] for c in chunks] if chunks else []
        })
    
    def _save_enhanced_data(self):
        """Salvamento de dados aprimorado"""
        # Em ambiente real, salvaria em disco
        self._log("Dados indexados salvos em memória")
    
    def fase5_interface_usuario(self):
        """Fase 5: Interface interativa melhorada"""
        self._log("=== FASE 5: INTERFACE INTERATIVA ===")
        
        print("\n" + "="*70)
        print("🤖 CHATBOT O GUARANI - VERSÃO MELHORADA")
        print("Assistente especializado na obra de José de Alencar")
        print(f"Método: {'Embeddings Semânticos' if self.use_embeddings else 'TF-IDF'}")
        print("Comandos: 'sair', 'historico', 'stats', 'ajuda'")
        print("="*70)
        
        while True:
            try:
                pergunta = input("\n💬 Sua pergunta: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                elif pergunta.lower() in ['historico', 'histórico', 'history']:
                    self.mostrar_historico_resumido()
                    continue
                elif pergunta.lower() in ['stats', 'estatísticas', 'estatisticas']:
                    self.mostrar_estatisticas()
                    continue
                elif pergunta.lower() in ['ajuda', 'help']:
                    self.mostrar_ajuda()
                    continue
                
                if not pergunta:
                    continue
                
                resposta = self.fase4_mecanismo_resposta(pergunta)
                print(f"\n🤖 {resposta}")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
    
    def mostrar_ajuda(self):
        """Mostra ajuda para o usuário"""
        help_text = """
        🆘 AJUDA - CHATBOT O GUARANI
        
        📝 Tipos de perguntas que funcionam bem:
        • Sobre personagens: "Quem é Peri?", "Fale sobre Cecília"
        • Sobre enredo: "Qual é a história?", "O que acontece no final?"
        • Sobre relacionamentos: "Qual a relação entre Peri e Ceci?"
        • Sobre cenário: "Onde se passa a história?"
        • Sobre temas: "Quais são os temas principais?"
        
        🎛️ Comandos especiais:
        • 'historico' - Ver histórico de conversas
        • 'stats' - Ver estatísticas do sistema
        • 'ajuda' - Mostrar esta ajuda
        • 'sair' - Encerrar o chatbot
        
        💡 Dicas para melhores respostas:
        • Seja específico em suas perguntas
        • Use nomes de personagens conhecidos
        • Reformule se a resposta não for satisfatória
        """
        print(help_text)
    
    def mostrar_historico_resumido(self):
        """Mostra histórico resumido das conversas"""
        if not self.conversation_history:
            print("📭 Nenhuma conversa no histórico ainda.")
            return
        
        print(f"\n📚 HISTÓRICO ({len(self.conversation_history)} conversas):")
        print("-" * 50)
        
        for i, conv in enumerate(self.conversation_history[-5:], 1):  # Últimas 5
            timestamp = conv['timestamp'].strftime("%H:%M")
            print(f"{i}. [{timestamp}] {conv['pergunta']}")
            print(f"   Similarity: {conv['similaridade_max']:.3f} | Chunks: {conv['chunks_usados']}")
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas detalhadas do sistema"""
        print(f"\n📊 ESTATÍSTICAS DO SISTEMA")
        print("=" * 40)
        print(f"Chunks de texto: {len(self.text_chunks)}")
        print(f"Método de vetorização: {'Embeddings' if self.use_embeddings else 'TF-IDF'}")
        print(f"Threshold de similaridade: {self.similarity_threshold}")
        print(f"Tamanho dos chunks: {self.chunk_size} palavras")
        print(f"Sobreposição: {self.overlap * 100}%")
        print(f"Consultas realizadas: {len(self.conversation_history)}")
        
        if self.performance_metrics:
            tempos = [m['tempo_processamento'] for m in self.performance_metrics]
            print(f"Tempo médio de resposta: {np.mean(tempos):.3f}s")
        
        if self.conversation_history:
            similaridades = [c['similaridade_max'] for c in self.conversation_history]
            print(f"Similaridade média: {np.mean(similaridades):.3f}")
    
    def executar_sistema_completo(self):
        """Executa todas as fases do sistema com tratamento de erros"""
        try:
            # Fase 1: Preparação
            if not self.fase1_preparar_ambiente():
                raise Exception("Falha na preparação do ambiente")
            
            # Fase 2: Processamento
            chunks = self.fase2_processar_dados()
            if not chunks:
                raise Exception("Falha no processamento dos dados")
            
            # Fase 3: Indexação
            if not self.fase3_armazenar_indexar(chunks):
                raise Exception("Falha na indexação")
            
            self._log("✅ Sistema inicializado com sucesso!")
            
            # Teste automático
            self.executar_testes_automaticos()
            
            return True
            
        except Exception as e:
            self._log(f"❌ Erro na execução: {e}")
            return False
    
    def executar_testes_automaticos(self):
        """Executa testes automáticos abrangentes"""
        perguntas_teste = [
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Qual é o enredo do livro?",
            "Quem são os personagens principais?",
            "Onde se passa a história?",
            "Qual a relação entre Peri e Ceci?",
            "Quem é Dom Antônio de Mariz?",
            "O que são os aimorés?",
            "Quando foi publicado O Guarani?",
            "Quais são os temas da obra?"
        ]
        
        print("\n" + "="*60)
        print("🧪 EXECUTANDO TESTES AUTOMÁTICOS")
        print("="*60)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n🔍 Teste {i}/10: {pergunta}")
            
            start_time = datetime.now()
            resposta = self.fase4_mecanismo_resposta(pergunta)
            tempo = (datetime.now() - start_time).total_seconds()
            
            # Análise da qualidade da resposta
            ultimo_historico = self.conversation_history[-1]
            qualidade = self._avaliar_qualidade_resposta(ultimo_historico)
            
            resultados.append({
                'pergunta': pergunta,
                'tempo': tempo,
                'similaridade': ultimo_historico['similaridade_max'],
                'chunks': ultimo_historico['chunks_usados'],
                'qualidade': qualidade
            })
            
            print(f"⏱️  Tempo: {tempo:.3f}s | Similaridade: {ultimo_historico['similaridade_max']:.3f} | {qualidade}")
            print(f"📝 Resposta: {resposta[:100]}...")
        
        # Relatório final
        self._gerar_relatorio_testes(resultados)
    
    def _avaliar_qualidade_resposta(self, historico: Dict) -> str:
        """Avalia a qualidade da resposta baseada em métricas"""
        sim = historico['similaridade_max']
        chunks = historico['chunks_usados']
        
        if sim > 0.4 and chunks > 0:
            return "🟢 Excelente"
        elif sim > 0.25 and chunks > 0:
            return "🟡 Boa"
        elif sim > 0.15 and chunks > 0:
            return "🟠 Regular"
        else:
            return "🔴 Ruim"
    
    def _gerar_relatorio_testes(self, resultados: List[Dict]):
        """Gera relatório final dos testes"""
        print("\n" + "="*60)
        print("📋 RELATÓRIO FINAL DOS TESTES")
        print("="*60)
        
        tempos = [r['tempo'] for r in resultados]
        similaridades = [r['similaridade'] for r in resultados]
        
        print(f"📊 Métricas Gerais:")
        print(f"   • Tempo médio: {np.mean(tempos):.3f}s")
        print(f"   • Tempo máximo: {np.max(tempos):.3f}s")
        print(f"   • Similaridade média: {np.mean(similaridades):.3f}")
        print(f"   • Similaridade mínima: {np.min(similaridades):.3f}")
        
        qualidades = [r['qualidade'] for r in resultados]
        excelentes = qualidades.count("🟢 Excelente")
        boas = qualidades.count("🟡 Boa")
        regulares = qualidades.count("🟠 Regular")
        ruins = qualidades.count("🔴 Ruim")
        
        print(f"\n🎯 Qualidade das Respostas:")
        print(f"   • Excelentes: {excelentes}/10 ({excelentes*10}%)")
        print(f"   • Boas: {boas}/10 ({boas*10}%)")
        print(f"   • Regulares: {regulares}/10 ({regulares*10}%)")
        print(f"   • Ruins: {ruins}/10 ({ruins*10}%)")
        
        # Recomendações
        if np.mean(similaridades) < 0.2:
            print("\n⚠️  Recomendação: Similaridades baixas. Considere:")
            print("    - Verificar se o texto está carregado corretamente")
            print("    - Ajustar threshold de similaridade")
            print("    - Melhorar o pré-processamento")


def main():
    """Função principal melhorada"""
    print("🚀 Iniciando Chatbot O Guarani - Versão Melhorada")
    print("=" * 60)
    
    chatbot = GuaraniChatbotImproved()
    
    if chatbot.executar_sistema_completo():
        print("\n✅ Sistema inicializado com sucesso!")
        
        # Menu de opções
        while True:
            print("\n🎯 OPÇÕES DISPONÍVEIS:")
            print("1. 💬 Iniciar chat interativo")
            print("2. 📊 Ver estatísticas do sistema")
            print("3. 📚 Ver histórico completo")
            print("4. 🧪 Executar novos testes")
            print("5. 🚪 Sair")
            
            try:
                opcao = input("\nEscolha uma opção (1-5): ").strip()
                
                if opcao == '1':
                    chatbot.fase5_interface_usuario()
                elif opcao == '2':
                    chatbot.mostrar_estatisticas()
                elif opcao == '3':
                    chatbot.mostrar_historico_resumido()
                elif opcao == '4':
                    chatbot.executar_testes_automaticos()
                elif opcao == '5':
                    print("👋 Encerrando sistema...")
                    break
                else:
                    print("❌ Opção inválida. Tente novamente.")
                    
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
    else:
        print("❌ Falha na inicialização do sistema")

if __name__ == "__main__":
    main()