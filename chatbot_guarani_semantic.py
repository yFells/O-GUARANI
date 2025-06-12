#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot "O Guarani" - Versão com Compreensão Semântica
Usando sentence-transformers para embeddings semânticos
"""

import numpy as np
import re
import os
from datetime import datetime
from typing import List, Dict, Optional
import time
import pickle
from pathlib import Path

# Importações para embeddings semânticos
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
    print("✅ Bibliotecas semânticas carregadas com sucesso!")
except ImportError as e:
    SEMANTIC_AVAILABLE = False
    print(f"❌ Erro ao importar bibliotecas semânticas: {e}")
    print("📦 Instale com: pip install sentence-transformers scikit-learn")

class GuaraniChatbotSemantico:
    """
    Chatbot O Guarani - Versão com Compreensão Semântica Avançada
    Usa sentence-transformers para entender o significado das perguntas
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani (Versão Semântica)")
        print("=" * 60)
        
        # Verificar dependências
        if not SEMANTIC_AVAILABLE:
            raise ImportError("❌ Bibliotecas semânticas não disponíveis. Execute: pip install sentence-transformers scikit-learn")
        
        # Configurações otimizadas
        self.chunk_size = 150
        self.overlap = 0.3
        self.similarity_threshold = 0.3  # Ajustado para embeddings (valores mais altos)
        self.top_chunks = 3
        
        # Estruturas de dados
        self.conversation_history = []
        self.processing_log = []
        self.performance_metrics = []
        self.text_chunks = []
        self.chunk_sentences = []
        self.chunk_embeddings = None  # Para armazenar embeddings dos chunks
        
        # Configuração do modelo semântico
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.sentence_model = None
        self.embeddings_cache_file = "guarani_embeddings_cache.pkl"
        
        # Stop words expandidas (ainda úteis para pré-processamento)
        self.stop_words = {
            'a', 'o', 'e', 'de', 'da', 'do', 'em', 'um', 'uma', 'com', 'para',
            'por', 'que', 'se', 'na', 'no', 'ao', 'aos', 'as', 'os', 'mais',
            'mas', 'ou', 'ter', 'ser', 'estar', 'seu', 'sua', 'seus', 'suas',
            'foi', 'são', 'dos', 'das', 'pela', 'pelo', 'sobre', 'até', 'sem',
            'muito', 'bem', 'já', 'ainda', 'só', 'pode', 'tem', 'vai', 'vem',
            'ele', 'ela', 'eles', 'elas', 'isso', 'isto', 'aquilo', 'quando',
            'onde', 'como', 'porque', 'então', 'assim', 'aqui', 'ali', 'lá',
            'me', 'te', 'nos', 'vos', 'lhe', 'lhes', 'meu', 'teu', 'nosso'
        }
        
        # Carregar texto do arquivo
        self.texto_guarani = self._carregar_texto_arquivo()
        
        if not self.texto_guarani:
            raise Exception("Falha ao carregar o arquivo guarani.txt")
        
        # Inicializar modelo semântico
        self._inicializar_modelo_semantico()
        
        self._log("Sistema inicializado com sucesso")
    
    def _inicializar_modelo_semantico(self):
        """Inicializa o modelo de sentence transformers"""
        try:
            self._log("🧠 Carregando modelo de embeddings semânticos...")
            self._log(f"📦 Modelo: {self.model_name}")
            
            # Carregar modelo (primeira vez pode demorar para download)
            start_time = time.time()
            self.sentence_model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            
            self._log(f"✅ Modelo carregado em {load_time:.2f}s")
            
            # Teste rápido do modelo
            test_embedding = self.sentence_model.encode(["Teste de funcionamento"])
            self._log(f"📏 Dimensão dos embeddings: {test_embedding.shape[1]}")
            
        except Exception as e:
            self._log(f"❌ Erro ao carregar modelo: {e}")
            raise e
    
    def _carregar_texto_arquivo(self) -> str:
        """Carrega o texto de O Guarani do arquivo guarani.txt"""
        arquivo_path = "guarani.txt"
        
        try:
            self._log("Tentando carregar guarani.txt...")
            
            # Verificar se o arquivo existe
            if not os.path.exists(arquivo_path):
                self._log(f"❌ Arquivo {arquivo_path} não encontrado!")
                self._log("Criando arquivo de exemplo...")
                self._criar_arquivo_exemplo(arquivo_path)
                return self._carregar_arquivo_exemplo()
            
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(arquivo_path, 'r', encoding=encoding) as file:
                        texto = file.read().strip()
                        
                    if texto and len(texto) > 100:  # Verificar se o texto não está vazio
                        self._log(f"✅ Arquivo carregado com encoding {encoding}")
                        self._log(f"📄 Tamanho do texto: {len(texto)} caracteres")
                        self._log(f"📝 Primeiras 100 chars: {texto[:100]}...")
                        return texto
                    else:
                        self._log(f"⚠️ Arquivo vazio ou muito pequeno com encoding {encoding}")
                        
                except UnicodeDecodeError:
                    self._log(f"❌ Falha com encoding {encoding}")
                    continue
                except Exception as e:
                    self._log(f"❌ Erro ao ler com {encoding}: {e}")
                    continue
            
            # Se chegou aqui, todas as tentativas falharam
            self._log("❌ Falha ao carregar com todos os encodings testados")
            self._log("📝 Usando texto de exemplo...")
            return self._carregar_arquivo_exemplo()
            
        except Exception as e:
            self._log(f"❌ Erro crítico ao carregar arquivo: {e}")
            self._log("📝 Usando texto de exemplo...")
            return self._carregar_arquivo_exemplo()
    
    def _criar_arquivo_exemplo(self, arquivo_path: str):
        """Cria um arquivo de exemplo com texto básico de O Guarani"""
        texto_exemplo = """O Guarani é um romance indianista de José de Alencar, publicado em 1857. A narrativa se desenvolve no século XVII, durante o período colonial brasileiro, nas montanhas fluminenses próximas ao rio Paquequer.

Peri é o protagonista da obra, um índio goitacá de força hercúlea e lealdade inabalável. Ele é descrito como um guerreiro corajoso, de estatura imponente e caráter nobre. Peri demonstra uma devoção absoluta a Cecília (Ceci), filha do fidalgo português Dom Antônio de Mariz.

Cecília, chamada carinhosamente de Ceci, é uma jovem portuguesa de beleza singular e caráter doce. Ela é filha de Dom Antônio de Mariz e representa a pureza e a inocência feminina idealizadas pelo Romantismo.

Dom Antônio de Mariz é um nobre português, fidalgo da Casa Real, que se estabeleceu no Brasil após cometer um crime de honra em Portugal. Ele construiu um castelo fortificado nas margens do rio Paquequer.

Álvaro é um jovem português, primo de Cecília, que também habita o castelo. Ele encarna o ideal do cavaleiro medieval, sendo corajoso, nobre e apaixonado por Ceci.

Isabel é irmã de Cecília, uma jovem impetuosa e apaixonada. Ela se enamora de Álvaro, criando um triângulo amoroso que adiciona complexidade às relações familiares.

Os aimorés são a tribo indígena antagonista, inimigos mortais de Peri e de sua tribo goitacá. Eles representam o perigo constante que ameaça a segurança dos habitantes do castelo.

Loredano é um dos antagonistas da história, um aventureiro italiano que se infiltra no castelo com intenções malévolas. Ele planeja assassinar Dom Antônio e se apossar de suas riquezas.

A natureza brasileira desempenha papel fundamental na narrativa, sendo descrita com exuberância e riqueza de detalhes. Alencar retrata as florestas, rios e montanhas como cenário épico.

O romance explora temas centrais como o amor impossível entre raças diferentes, representado pela relação entre Peri e Ceci. A lealdade e o sacrifício são exemplificados pela devoção absoluta do índio à família Mariz."""
        
        try:
            with open(arquivo_path, 'w', encoding='utf-8') as file:
                file.write(texto_exemplo)
            self._log(f"✅ Arquivo de exemplo criado: {arquivo_path}")
        except Exception as e:
            self._log(f"❌ Erro ao criar arquivo de exemplo: {e}")
    
    def _carregar_arquivo_exemplo(self) -> str:
        """Retorna texto de exemplo quando o arquivo não pode ser carregado"""
        return """O Guarani é um romance indianista de José de Alencar, publicado em 1857. A narrativa se desenvolve no século XVII, durante o período colonial brasileiro, nas montanhas fluminenses próximas ao rio Paquequer.

Peri é o protagonista da obra, um índio goitacá de força hercúlea e lealdade inabalável. Ele é descrito como um guerreiro corajoso, de estatura imponente e caráter nobre. Peri demonstra uma devoção absoluta a Cecília (Ceci), filha do fidalgo português Dom Antônio de Mariz. Esta devoção representa o amor impossível entre duas raças distintas.

Cecília, chamada carinhosamente de Ceci, é uma jovem portuguesa de beleza singular e caráter doce. Ela é filha de Dom Antônio de Mariz e representa a pureza e a inocência feminina idealizadas pelo Romantismo. Ceci desenvolve sentimentos fraternais por Peri, vendo nele um protetor dedicado.

Dom Antônio de Mariz é um nobre português, fidalgo da Casa Real, que se estabeleceu no Brasil após cometer um crime de honra em Portugal. Ele construiu um castelo fortificado nas margens do rio Paquequer, onde vive com sua família. Dom Antônio é caracterizado como um homem honrado, mas marcado pelo passado.

Álvaro é um jovem português, primo de Cecília, que também habita o castelo. Ele encarna o ideal do cavaleiro medieval, sendo corajoso, nobre e apaixonado por Ceci. Álvaro representa a civilização europeia em contraste com a natureza selvagem de Peri.

Isabel é irmã de Cecília, uma jovem impetuosa e apaixonada. Ela se enamora de Álvaro, criando um triângulo amoroso que adiciona complexidade às relações familiares. Isabel possui um temperamento mais forte que sua irmã.

Os aimorés são a tribo indígena antagonista, inimigos mortais de Peri e de sua tribo goitacá. Eles representam o perigo constante que ameaça a segurança dos habitantes do castelo. Os aimorés são descritos como selvagens e canibais.

Loredano é um dos antagonistas da história, um aventureiro italiano que se infiltra no castelo com intenções malévolas. Ele planeja assassinar Dom Antônio e se apossar de suas riquezas, representando a traição e a vilania."""
    
    def _log(self, message: str):
        """Log seguro"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(f"📝 {log_entry}")
    
    def fase1_analisar_texto(self):
        """Fase 1: Análise do texto"""
        self._log("=== FASE 1: ANÁLISE DO TEXTO ===")
        
        if not self.texto_guarani:
            self._log("❌ Texto não carregado!")
            return False
        
        chars = len(self.texto_guarani)
        words = self.texto_guarani.split()
        sentences = self._segmentar_sentencas(self.texto_guarani)
        
        word_tokens = re.findall(r'\b\w+\b', self.texto_guarani.lower())
        unique_words = set(word_tokens)
        content_words = unique_words - self.stop_words
        
        self._log(f"Caracteres: {chars}")
        self._log(f"Palavras: {len(words)}")
        self._log(f"Sentenças: {len(sentences)}")
        self._log(f"Vocabulário único: {len(unique_words)}")
        self._log(f"Palavras de conteúdo: {len(content_words)}")
        
        return True
    
    def _segmentar_sentencas(self, texto: str) -> List[str]:
        """Segmentação robusta de sentenças"""
        # Limpeza inicial
        texto = re.sub(r'\n+', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        # Segmentação por pontuação
        sentences = re.split(r'[.!?]+', texto)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]
        
        return sentences
    
    def fase2_criar_chunks_e_embeddings(self):
        """Fase 2: Criação de chunks e geração de embeddings semânticos"""
        self._log("=== FASE 2: CRIAÇÃO DE CHUNKS E EMBEDDINGS ===")
        
        if not self.texto_guarani:
            self._log("❌ Texto não carregado!")
            return False
        
        if not self.sentence_model:
            self._log("❌ Modelo semântico não carregado!")
            return False
        
        # Criar chunks de texto
        sentences = self._segmentar_sentencas(self.texto_guarani)
        
        chunks = []
        chunk_sentences_map = []
        current_chunk_sentences = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            # Verificar se cabe no chunk atual
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
            else:
                # Finalizar chunk atual
                if current_chunk_sentences:
                    chunk_text = '. '.join(current_chunk_sentences) + '.'
                    chunks.append(chunk_text)
                    chunk_sentences_map.append(current_chunk_sentences.copy())
                
                # Aplicar sobreposição
                overlap_size = int(len(current_chunk_sentences) * self.overlap)
                if overlap_size > 0 and len(current_chunk_sentences) > overlap_size:
                    current_chunk_sentences = current_chunk_sentences[-overlap_size:]
                    current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_word_count = 0
                
                # Adicionar nova sentença
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
        
        # Finalizar último chunk
        if current_chunk_sentences:
            chunk_text = '. '.join(current_chunk_sentences) + '.'
            chunks.append(chunk_text)
            chunk_sentences_map.append(current_chunk_sentences.copy())
        
        self.text_chunks = chunks
        self.chunk_sentences = chunk_sentences_map
        
        # Verificar se existe cache de embeddings
        cache_exists = self._verificar_cache_embeddings()
        
        if cache_exists:
            self._log("📁 Cache de embeddings encontrado, carregando...")
            if self._carregar_cache_embeddings():
                self._log("✅ Embeddings carregados do cache!")
            else:
                self._log("❌ Falha ao carregar cache, gerando novos embeddings...")
                self._gerar_embeddings()
        else:
            self._log("🧠 Gerando embeddings semânticos dos chunks...")
            self._gerar_embeddings()
        
        # Estatísticas
        if chunks and self.chunk_embeddings is not None:
            chunk_sizes = [len(chunk.split()) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            self._log(f"Chunks criados: {len(chunks)}")
            self._log(f"Tamanho médio: {avg_size:.1f} palavras")
            self._log(f"Embeddings shape: {self.chunk_embeddings.shape}")
        
        return True
    
    def _verificar_cache_embeddings(self) -> bool:
        """Verifica se existe cache de embeddings válido"""
        try:
            if not os.path.exists(self.embeddings_cache_file):
                return False
            
            # Verificar se o cache não está muito antigo
            cache_time = os.path.getmtime(self.embeddings_cache_file)
            text_time = os.path.getmtime("guarani.txt") if os.path.exists("guarani.txt") else 0
            
            # Se o texto foi modificado depois do cache, invalidar
            if text_time > cache_time:
                self._log("⚠️ Texto modificado, cache inválido")
                return False
            
            return True
            
        except Exception as e:
            self._log(f"❌ Erro ao verificar cache: {e}")
            return False
    
    def _carregar_cache_embeddings(self) -> bool:
        """Carrega embeddings do cache"""
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verificar integridade do cache
            if ('embeddings' in cache_data and 
                'chunks' in cache_data and 
                len(cache_data['chunks']) == len(self.text_chunks)):
                
                # Verificar se os chunks são os mesmos
                if cache_data['chunks'] == self.text_chunks:
                    self.chunk_embeddings = cache_data['embeddings']
                    return True
                else:
                    self._log("⚠️ Chunks diferentes do cache, regenerando...")
                    return False
            else:
                self._log("⚠️ Cache inválido ou corrompido")
                return False
                
        except Exception as e:
            self._log(f"❌ Erro ao carregar cache: {e}")
            return False
    
    def _salvar_cache_embeddings(self):
        """Salva embeddings no cache"""
        try:
            cache_data = {
                'embeddings': self.chunk_embeddings,
                'chunks': self.text_chunks,
                'model_name': self.model_name,
                'timestamp': datetime.now()
            }
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self._log(f"💾 Cache salvo: {self.embeddings_cache_file}")
            
        except Exception as e:
            self._log(f"❌ Erro ao salvar cache: {e}")
    
    def _gerar_embeddings(self):
        """Gera embeddings semânticos para todos os chunks"""
        try:
            start_time = time.time()
            
            # Gerar embeddings para todos os chunks de uma vez (mais eficiente)
            self._log(f"🧠 Processando {len(self.text_chunks)} chunks...")
            
            self.chunk_embeddings = self.sentence_model.encode(
                self.text_chunks,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32  # Processamento em lotes para eficiência
            )
            
            generation_time = time.time() - start_time
            self._log(f"✅ Embeddings gerados em {generation_time:.2f}s")
            self._log(f"📊 Shape dos embeddings: {self.chunk_embeddings.shape}")
            
            # Salvar no cache
            self._salvar_cache_embeddings()
            
        except Exception as e:
            self._log(f"❌ Erro ao gerar embeddings: {e}")
            raise e
    
    def calcular_similaridade_semantica(self, pergunta: str, chunks: Optional[List[str]] = None) -> List[float]:
        """Calcula similaridade semântica usando embeddings"""
        try:
            if chunks is None:
                chunks = self.text_chunks
            
            if self.chunk_embeddings is None:
                self._log("❌ Embeddings não carregados!")
                return [0.0] * len(chunks)
            
            # Gerar embedding da pergunta
            question_embedding = self.sentence_model.encode([pergunta], convert_to_numpy=True)
            
            # Calcular similaridade coseno
            similarities = cosine_similarity(question_embedding, self.chunk_embeddings)[0]
            
            # Converter para lista de floats
            similarities = [float(sim) for sim in similarities]
            
            return similarities
            
        except Exception as e:
            self._log(f"❌ Erro no cálculo de similaridade semântica: {e}")
            return [0.0] * len(chunks if chunks else self.text_chunks)
    
    def fase3_responder_pergunta(self, pergunta: str) -> str:
        """Fase 3: Resposta à pergunta usando similaridade semântica"""
        start_time = time.time()
        self._log(f"=== CONSULTA SEMÂNTICA: {pergunta} ===")
        
        if not self.text_chunks:
            return "❌ Sistema não processado. Execute as fases anteriores."
        
        if self.chunk_embeddings is None:
            return "❌ Embeddings não carregados. Execute a Fase 2."
        
        try:
            # Calcular similaridades semânticas
            similarities = self.calcular_similaridade_semantica(pergunta)
            
            # Verificar se temos similaridades válidas
            if not similarities:
                return "❌ Erro no cálculo de similaridades semânticas."
            
            # Criar resultados de forma segura
            chunk_results = []
            for i, sim in enumerate(similarities):
                chunk_results.append({
                    'chunk_id': i,
                    'chunk': self.text_chunks[i],
                    'similarity': float(sim),
                    'sentences': self.chunk_sentences[i] if i < len(self.chunk_sentences) else []
                })
            
            # Ordenar por similaridade
            chunk_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Estatísticas seguras
            max_sim = chunk_results[0]['similarity'] if chunk_results else 0.0
            mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
            
            self._log(f"Similaridade semântica máxima: {max_sim:.3f}")
            self._log(f"Similaridade semântica média: {mean_sim:.3f}")
            
            # Filtrar chunks relevantes (threshold mais alto para embeddings)
            relevant_chunks = []
            for chunk in chunk_results:
                if chunk['similarity'] >= self.similarity_threshold:
                    relevant_chunks.append(chunk)
            
            self._log(f"Chunks semanticamente relevantes: {len(relevant_chunks)}")
            
            # Gerar resposta
            if not relevant_chunks:
                response = self._resposta_nao_encontrada_semantica(pergunta, max_sim)
            else:
                response = self._gerar_resposta_semantica(pergunta, relevant_chunks[:self.top_chunks])
            
            # Métricas
            processing_time = time.time() - start_time
            self.performance_metrics.append({
                'pergunta': pergunta,
                'tempo': processing_time,
                'max_similarity': max_sim,
                'chunks_relevantes': len(relevant_chunks),
                'metodo': 'semantico'
            })
            
            # Histórico
            self.conversation_history.append({
                'pergunta': pergunta,
                'resposta': response,
                'similaridade_max': max_sim,
                'chunks_usados': len(relevant_chunks),
                'tempo_resposta': processing_time,
                'metodo': 'semantico',
                'timestamp': datetime.now()
            })
            
            self._log(f"Resposta semântica gerada em {processing_time:.3f}s")
            return response
            
        except Exception as e:
            error_msg = f"❌ Erro inesperado na análise semântica: {e}"
            self._log(error_msg)
            return error_msg
    
    def _resposta_nao_encontrada_semantica(self, pergunta: str, max_sim: float) -> str:
        """Resposta quando não encontra informações semanticamente relevantes"""
        base_msg = "Não encontrei informações semanticamente relevantes sobre sua pergunta no texto de 'O Guarani'."
        
        if max_sim > 0.2:
            suggestion = "\n\n💡 Tente reformular usando termos mais específicos ou sinônimos."
        elif max_sim > 0.1:
            suggestion = "\n\n💡 Use nomes de personagens ou conceitos centrais da obra."
        else:
            suggestion = "\n\n💡 Sua pergunta pode estar completamente fora do escopo da obra."
        
        examples = """
\n📝 Exemplos de perguntas que funcionam bem com análise semântica:
• "Como é a personalidade de Peri?"
• "Qual o sentimento entre Peri e Cecília?"
• "Descreva o conflito principal da obra"
• "Quais são os antagonistas da história?"
• "Como é retratada a natureza brasileira?"
• "Qual o papel da família Mariz?"
• "Quais são os temas do romance?"
• "Como é caracterizado o amor impossível?"
"""
        
        confidence = f"\n\n🔴 Similaridade semântica baixa (máxima: {max_sim:.3f})"
        
        return base_msg + suggestion + examples + confidence
    
    def _gerar_resposta_semantica(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta usando análise semântica"""
        if not chunks:
            return self._resposta_nao_encontrada_semantica(pergunta, 0.0)
        
        try:
            best_chunk = chunks[0]
            
            # Para embeddings semânticos, vamos usar o chunk completo mais relevante
            # pois a similaridade já captura o significado geral
            
            if len(chunks) == 1:
                main_content = chunks[0]['chunk']
                intro = "Com base na análise semântica de 'O Guarani':\n\n"
            else:
                # Combinar os chunks mais relevantes semanticamente
                combined_chunks = []
                total_length = 0
                similarity_scores = []
                
                for chunk in chunks[:3]:  # Máximo 3 chunks semanticamente relevantes
                    chunk_text = chunk['chunk']
                    if total_length + len(chunk_text) < 800:  # Limite um pouco maior para semântica
                        combined_chunks.append(chunk_text)
                        similarity_scores.append(chunk['similarity'])
                        total_length += len(chunk_text)
                    else:
                        break
                
                main_content = "\n\n".join(combined_chunks)
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                intro = f"Combinando informações semanticamente relevantes de 'O Guarani' (similaridade média: {avg_similarity:.3f}):\n\n"
            
            # Truncar se muito longo, mas manter mais contexto para análise semântica
            if len(main_content) > 700:
                main_content = main_content[:700] + "..."
            
            confidence = self._calcular_confianca_semantica(best_chunk['similarity'])
            return intro + main_content + "\n\n" + confidence
            
        except Exception as e:
            self._log(f"Erro na geração de resposta semântica: {e}")
            return f"❌ Erro ao gerar resposta semântica: {e}"
    
    def _calcular_confianca_semantica(self, similarity: float) -> str:
        """Calcula indicador de confiança para similaridade semântica"""
        try:
            sim = float(similarity)
            if sim > 0.7:
                return "🟢 Confiança semântica muito alta"
            elif sim > 0.5:
                return "🟢 Confiança semântica alta"
            elif sim > 0.4:
                return "🟡 Confiança semântica moderada"
            elif sim > 0.3:
                return "🟠 Confiança semântica baixa"
            else:
                return "🔴 Confiança semântica muito baixa"
        except:
            return "⚠️ Confiança semântica indeterminada"
    
    def executar_sistema_completo(self):
        """Executa todas as fases do sistema semântico"""
        try:
            self._log("🚀 EXECUTANDO SISTEMA SEMÂNTICO COMPLETO")
            
            if not self.fase1_analisar_texto():
                raise Exception("Erro na Fase 1")
            
            if not self.fase2_criar_chunks_e_embeddings():
                raise Exception("Erro na Fase 2 (Embeddings)")
            
            self._log("✅ Sistema semântico pronto para consultas!")
            return True
            
        except Exception as e:
            self._log(f"❌ Erro na execução: {e}")
            return False
    
    def comparar_metodos(self, pergunta: str) -> Dict:
        """Compara método semântico com método tradicional (Jaccard)"""
        self._log(f"🔬 COMPARANDO MÉTODOS: {pergunta}")
        
        try:
            # Método semântico
            start_semantic = time.time()
            similarities_semantic = self.calcular_similaridade_semantica(pergunta)
            time_semantic = time.time() - start_semantic
            
            # Método Jaccard (tradicional) para comparação
            start_jaccard = time.time()
            similarities_jaccard = []
            for chunk in self.text_chunks:
                sim = self._calcular_jaccard_simples(pergunta, chunk)
                similarities_jaccard.append(sim)
            time_jaccard = time.time() - start_jaccard
            
            # Análise comparativa
            max_semantic = max(similarities_semantic) if similarities_semantic else 0
            max_jaccard = max(similarities_jaccard) if similarities_jaccard else 0
            
            # Contar chunks relevantes para cada método
            relevant_semantic = sum(1 for s in similarities_semantic if s >= 0.3)
            relevant_jaccard = sum(1 for s in similarities_jaccard if s >= 0.15)
            
            comparison = {
                'pergunta': pergunta,
                'semantic': {
                    'max_similarity': max_semantic,
                    'relevant_chunks': relevant_semantic,
                    'time': time_semantic,
                    'threshold': 0.3
                },
                'jaccard': {
                    'max_similarity': max_jaccard,
                    'relevant_chunks': relevant_jaccard,
                    'time': time_jaccard,
                    'threshold': 0.15
                }
            }
            
            self._log(f"📊 Semântico: {max_semantic:.3f} sim, {relevant_semantic} chunks, {time_semantic:.3f}s")
            self._log(f"📊 Jaccard: {max_jaccard:.3f} sim, {relevant_jaccard} chunks, {time_jaccard:.3f}s")
            
            return comparison
            
        except Exception as e:
            self._log(f"❌ Erro na comparação: {e}")
            return {}
    
    def _calcular_jaccard_simples(self, pergunta: str, texto: str) -> float:
        """Implementação simples de Jaccard para comparação"""
        try:
            pergunta_words = set(re.findall(r'\b\w+\b', pergunta.lower()))
            texto_words = set(re.findall(r'\b\w+\b', texto.lower()))
            
            # Remover stop words
            pergunta_words = pergunta_words - self.stop_words
            texto_words = texto_words - self.stop_words
            
            if not pergunta_words or not texto_words:
                return 0.0
            
            intersection = len(pergunta_words & texto_words)
            union = len(pergunta_words | texto_words)
            
            return intersection / union if union > 0 else 0.0
            
        except:
            return 0.0
    
    def executar_testes_comparativos(self):
        """Executa testes comparando método semântico com Jaccard"""
        perguntas_teste = [
            # Testes diretos (devem funcionar bem com ambos)
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Quem é Dom Antônio de Mariz?",
            
            # Testes semânticos (devem funcionar melhor com embeddings)
            "Como é a personalidade do protagonista?",
            "Qual é o sentimento entre os personagens principais?",
            "Descreva o conflito central da obra",
            "Como é retratado o amor impossível?",
            "Qual o papel da natureza na narrativa?",
            "Fale sobre os antagonistas da história",
            "Como são caracterizados os valores europeus?",
            "Qual a importância do castelo na obra?",
            
            # Testes de sinônimos (embeddings devem ser superiores)
            "Quem é o herói da história?",  # Peri
            "Fale sobre a donzela da obra",  # Cecília
            "Descreva os inimigos dos protagonistas",  # Aimorés
            "Como é a floresta na narrativa?",  # Natureza
            
            # Testes negativos
            "Como fazer um bolo?",
            "Qual a capital da França?"
        ]
        
        print(f"\n🧪 EXECUTANDO TESTES COMPARATIVOS ({len(perguntas_teste)} perguntas)")
        print("=" * 80)
        print("🔬 Comparando Método Semântico vs Método Jaccard")
        print("=" * 80)
        
        resultados_comparativos = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n📋 Teste {i:2d}/{len(perguntas_teste)}: {pergunta}")
            
            try:
                comparison = self.comparar_metodos(pergunta)
                
                if comparison:
                    resultados_comparativos.append(comparison)
                    
                    sem = comparison['semantic']
                    jac = comparison['jaccard']
                    
                    print(f"   🧠 Semântico: {sem['max_similarity']:.3f} | {sem['relevant_chunks']} chunks | {sem['time']:.3f}s")
                    print(f"   📝 Jaccard:   {jac['max_similarity']:.3f} | {jac['relevant_chunks']} chunks | {jac['time']:.3f}s")
                    
                    # Determinar qual método foi melhor
                    if sem['max_similarity'] > jac['max_similarity']:
                        print(f"   🏆 Semântico venceu!")
                    elif jac['max_similarity'] > sem['max_similarity']:
                        print(f"   🏆 Jaccard venceu!")
                    else:
                        print(f"   🤝 Empate!")
                
            except Exception as e:
                print(f"   ❌ ERRO: {e}")
        
        # Relatório comparativo final
        self._relatorio_comparativo(resultados_comparativos)
        return resultados_comparativos
    
    def _relatorio_comparativo(self, resultados: List[Dict]):
        """Gera relatório comparativo entre os métodos"""
        print(f"\n📋 RELATÓRIO COMPARATIVO")
        print("=" * 60)
        
        if not resultados:
            print("❌ Nenhum resultado para analisar")
            return
        
        try:
            # Métricas semânticas
            semantic_scores = [r['semantic']['max_similarity'] for r in resultados]
            semantic_times = [r['semantic']['time'] for r in resultados]
            semantic_chunks = [r['semantic']['relevant_chunks'] for r in resultados]
            
            # Métricas Jaccard
            jaccard_scores = [r['jaccard']['max_similarity'] for r in resultados]
            jaccard_times = [r['jaccard']['time'] for r in resultados]
            jaccard_chunks = [r['jaccard']['relevant_chunks'] for r in resultados]
            
            # Calcular médias
            sem_avg_score = sum(semantic_scores) / len(semantic_scores)
            jac_avg_score = sum(jaccard_scores) / len(jaccard_scores)
            sem_avg_time = sum(semantic_times) / len(semantic_times)
            jac_avg_time = sum(jaccard_times) / len(jaccard_times)
            sem_avg_chunks = sum(semantic_chunks) / len(semantic_chunks)
            jac_avg_chunks = sum(jaccard_chunks) / len(jaccard_chunks)
            
            print(f"📊 COMPARAÇÃO DE PERFORMANCE:")
            print(f"   🧠 Método Semântico:")
            print(f"      • Similaridade média: {sem_avg_score:.3f}")
            print(f"      • Tempo médio: {sem_avg_time:.3f}s")
            print(f"      • Chunks relevantes (média): {sem_avg_chunks:.1f}")
            
            print(f"   📝 Método Jaccard:")
            print(f"      • Similaridade média: {jac_avg_score:.3f}")
            print(f"      • Tempo médio: {jac_avg_time:.3f}s")
            print(f"      • Chunks relevantes (média): {jac_avg_chunks:.1f}")
            
            # Análise de vitórias
            semantic_wins = 0
            jaccard_wins = 0
            ties = 0
            
            for r in resultados:
                sem_score = r['semantic']['max_similarity']
                jac_score = r['jaccard']['max_similarity']
                
                if sem_score > jac_score:
                    semantic_wins += 1
                elif jac_score > sem_score:
                    jaccard_wins += 1
                else:
                    ties += 1
            
            total = len(resultados)
            print(f"\n🏆 ANÁLISE DE VITÓRIAS:")
            print(f"   • Semântico: {semantic_wins}/{total} ({semantic_wins/total*100:.1f}%)")
            print(f"   • Jaccard: {jaccard_wins}/{total} ({jaccard_wins/total*100:.1f}%)")
            print(f"   • Empates: {ties}/{total} ({ties/total*100:.1f}%)")
            
            # Conclusão
            print(f"\n📈 CONCLUSÃO:")
            if semantic_wins > jaccard_wins:
                advantage = ((semantic_wins - jaccard_wins) / total) * 100
                print(f"   ✅ Método Semântico superior em {advantage:.1f}% dos casos")
                print(f"   🎯 Melhor para perguntas conceituais e sinônimos")
            elif jaccard_wins > semantic_wins:
                advantage = ((jaccard_wins - semantic_wins) / total) * 100
                print(f"   ✅ Método Jaccard superior em {advantage:.1f}% dos casos")
                print(f"   🎯 Melhor para correspondências exatas de palavras")
            else:
                print(f"   🤝 Métodos equivalentes na maioria dos casos")
            
            print(f"   ⏱️ Diferença de tempo: {sem_avg_time - jac_avg_time:.3f}s (Semântico - Jaccard)")
            
        except Exception as e:
            print(f"❌ Erro no relatório comparativo: {e}")
    
    def verificar_arquivo_info(self):
        """Mostra informações sobre o arquivo carregado"""
        print(f"\n📁 INFORMAÇÕES DO ARQUIVO")
        print("=" * 40)
        
        arquivo_path = "guarani.txt"
        
        try:
            if os.path.exists(arquivo_path):
                file_stats = os.stat(arquivo_path)
                file_size = file_stats.st_size
                mod_time = datetime.fromtimestamp(file_stats.st_mtime)
                
                print(f"📄 Arquivo: {arquivo_path}")
                print(f"📏 Tamanho: {file_size} bytes")
                print(f"📅 Modificado em: {mod_time.strftime('%d/%m/%Y %H:%M:%S')}")
                print(f"✅ Status: Encontrado")
            else:
                print(f"📄 Arquivo: {arquivo_path}")
                print(f"❌ Status: Não encontrado")
                print(f"💡 O sistema criará um arquivo de exemplo se necessário")
            
            # Info do cache de embeddings
            if os.path.exists(self.embeddings_cache_file):
                cache_stats = os.stat(self.embeddings_cache_file)
                cache_size = cache_stats.st_size
                cache_time = datetime.fromtimestamp(cache_stats.st_mtime)
                print(f"\n💾 Cache de embeddings: {self.embeddings_cache_file}")
                print(f"📏 Tamanho do cache: {cache_size:,} bytes")
                print(f"📅 Gerado em: {cache_time.strftime('%d/%m/%Y %H:%M:%S')}")
            else:
                print(f"\n💾 Cache de embeddings: Não existe")
            
            if self.texto_guarani:
                words = len(self.texto_guarani.split())
                lines = len(self.texto_guarani.split('\n'))
                chars = len(self.texto_guarani)
                
                print(f"\n📊 CONTEÚDO CARREGADO:")
                print(f"   • Caracteres: {chars:,}")
                print(f"   • Palavras: {words:,}")
                print(f"   • Linhas: {lines:,}")
                print(f"   • Primeiros 150 chars: {self.texto_guarani[:150]}...")
            
            # Info do modelo semântico
            if self.sentence_model:
                print(f"\n🧠 MODELO SEMÂNTICO:")
                print(f"   • Modelo: {self.model_name}")
                print(f"   • Status: ✅ Carregado")
                if self.chunk_embeddings is not None:
                    print(f"   • Embeddings: {self.chunk_embeddings.shape}")
                else:
                    print(f"   • Embeddings: ❌ Não gerados")
            else:
                print(f"\n🧠 MODELO SEMÂNTICO: ❌ Não carregado")
                
        except Exception as e:
            print(f"❌ Erro ao verificar arquivo: {e}")
    
    def mostrar_estatisticas(self):
        """Estatísticas do sistema semântico"""
        print(f"\n📊 ESTATÍSTICAS DO SISTEMA SEMÂNTICO")
        print("=" * 50)
        
        try:
            print(f"📝 Chunks: {len(self.text_chunks)}")
            print(f"🔧 Threshold semântico: {self.similarity_threshold}")
            print(f"📏 Tamanho chunks: {self.chunk_size} palavras")
            print(f"🔄 Sobreposição: {self.overlap * 100}%")
            print(f"💬 Consultas: {len(self.conversation_history)}")
            print(f"🧠 Modelo: {self.model_name}")
            print(f"🛠️ Método: Embeddings semânticos")
            
            if self.chunk_embeddings is not None:
                print(f"📊 Embeddings: {self.chunk_embeddings.shape}")
                print(f"💾 Cache: {self.embeddings_cache_file}")
            
            if self.texto_guarani:
                words = len(self.texto_guarani.split())
                chars = len(self.texto_guarani)
                print(f"📄 Texto: {chars:,} chars, {words:,} palavras")
            
            if self.performance_metrics:
                tempos = [float(m.get('tempo', 0)) for m in self.performance_metrics if m.get('tempo')]
                if tempos:
                    print(f"⏱️ Tempo médio: {sum(tempos)/len(tempos):.3f}s")
            
            if self.conversation_history:
                similarities = [float(c.get('similaridade_max', 0)) for c in self.conversation_history if c.get('similaridade_max')]
                if similarities:
                    print(f"📈 Similaridade semântica média: {sum(similarities)/len(similarities):.3f}")
                    print(f"📈 Similaridade semântica máxima: {max(similarities):.3f}")
                    
                # Contar métodos usados
                metodos = [c.get('metodo', 'indefinido') for c in self.conversation_history]
                semanticos = metodos.count('semantico')
                print(f"🧠 Consultas semânticas: {semanticos}/{len(self.conversation_history)}")
                    
        except Exception as e:
            print(f"❌ Erro nas estatísticas: {e}")
    
    def interface_chat(self):
        """Interface de chat com capacidades semânticas"""
        print(f"\n🤖 CHATBOT O GUARANI - CHAT SEMÂNTICO INTERATIVO")
        print("=" * 60)
        print("Comandos especiais:")
        print("  • 'sair' - Encerrar chat")
        print("  • 'stats' - Ver estatísticas")
        print("  • 'comparar' - Comparar métodos na última pergunta")
        print("  • 'teste' - Executar testes comparativos")
        print("  • 'help' - Mostrar ajuda")
        print("  • 'arquivo' - Info sobre arquivo")
        print("  • 'cache' - Limpar cache de embeddings")
        print("=" * 60)
        
        ultima_pergunta = ""
        
        while True:
            try:
                pergunta = input("\n💬 Sua pergunta: ").strip()
                
                if not pergunta:
                    print("⚠️ Digite uma pergunta ou comando.")
                    continue
                
                if pergunta.lower() in ['sair', 'exit', 'quit', 'tchau']:
                    print("👋 Até logo!")
                    break
                elif pergunta.lower() in ['stats', 'estatisticas', 'estatísticas']:
                    self.mostrar_estatisticas()
                    continue
                elif pergunta.lower() in ['comparar', 'compare'] and ultima_pergunta:
                    self.comparar_metodos(ultima_pergunta)
                    continue
                elif pergunta.lower() in ['teste', 'testes', 'test']:
                    self.executar_testes_comparativos()
                    continue
                elif pergunta.lower() in ['help', 'ajuda', '?']:
                    self._mostrar_ajuda_semantica()
                    continue
                elif pergunta.lower() in ['arquivo', 'file', 'info']:
                    self.verificar_arquivo_info()
                    continue
                elif pergunta.lower() in ['cache', 'clear', 'limpar']:
                    self._limpar_cache()
                    continue
                
                # Processar pergunta normal
                try:
                    resposta = self.fase3_responder_pergunta(pergunta)
                    print(f"\n🤖 {resposta}")
                    ultima_pergunta = pergunta
                except Exception as e:
                    print(f"\n❌ Erro ao processar pergunta: {e}")
                    print("Tente reformular sua pergunta.")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                print("Digite 'sair' para encerrar ou continue tentando.")
    
    def _limpar_cache(self):
        """Limpa o cache de embeddings"""
        try:
            if os.path.exists(self.embeddings_cache_file):
                os.remove(self.embeddings_cache_file)
                print("✅ Cache de embeddings removido!")
                print("⚠️ Embeddings serão regenerados na próxima execução")
            else:
                print("ℹ️ Nenhum cache encontrado")
        except Exception as e:
            print(f"❌ Erro ao limpar cache: {e}")
    
    def _mostrar_ajuda_semantica(self):
        """Mostra ajuda específica para o sistema semântico"""
        help_text = """
🆘 AJUDA - CHATBOT O GUARANI SEMÂNTICO

🧠 ANÁLISE SEMÂNTICA:
   • O sistema usa embeddings para entender SIGNIFICADO
   • Funciona bem com sinônimos e conceitos relacionados
   • Não depende apenas de palavras exatas

📁 ARQUIVO:
   • Carrega texto do arquivo 'guarani.txt'
   • Gera cache de embeddings para velocidade
   • Use 'arquivo' para informações detalhadas

📝 PERGUNTAS QUE FUNCIONAM MUITO BEM:

🎭 Conceituais (força do sistema semântico):
   • "Como é a personalidade do protagonista?"
   • "Qual o sentimento entre os personagens?"
   • "Descreva o conflito central da obra"
   • "Como é retratado o amor impossível?"
   • "Qual o papel da natureza na narrativa?"

🧑 Usando sinônimos:
   • "Quem é o herói da história?" (= Peri)
   • "Fale sobre a donzela" (= Cecília)
   • "Descreva os inimigos" (= aimorés)
   • "Como é a floresta?" (= natureza)

💕 Relacionamentos e temas:
   • "Qual a devoção de Peri?"
   • "Como são os valores europeus?"
   • "Quais são os antagonistas?"
   • "Fale sobre lealdade na obra"

🏰 Contextuais:
   • "Onde acontece a história?"
   • "Como é o ambiente da obra?"
   • "Qual a época retratada?"

💡 VANTAGENS SEMÂNTICAS:
   • Entende sinônimos e conceitos relacionados
   • Não precisa de palavras exatas
   • Melhor para perguntas conceituais
   • Compreende contexto e significado

🔧 COMANDOS ESPECIAIS:
   • 'comparar' - Compara último resultado com Jaccard
   • 'teste' - Teste comparativo completo
   • 'cache' - Limpar cache de embeddings
   • 'stats' - Estatísticas do sistema semântico

⚡ PERFORMANCE:
   • Primeira execução: mais lenta (gera embeddings)
   • Execuções seguintes: rápida (usa cache)
   • Melhor qualidade semântica que métodos tradicionais
        """
        print(help_text)

def main():
    """Função principal do sistema semântico"""
    print("🎯 CHATBOT O GUARANI - VERSÃO SEMÂNTICA AVANÇADA")
    print("=" * 70)
    print("🧠 Esta versão usa embeddings semânticos para compreensão avançada")
    print("📦 Requer: sentence-transformers, scikit-learn")
    print("⚡ Primeira execução pode ser mais lenta (download do modelo)")
    print()
    
    try:
        chatbot = GuaraniChatbotSemantico()
        
        # Mostrar informações do arquivo carregado
        chatbot.verificar_arquivo_info()
        
        if chatbot.executar_sistema_completo():
            print("\n✅ Sistema semântico inicializado com sucesso!")
            print("🧠 Compreensão semântica ativada!")
            print("📁 Texto carregado do arquivo guarani.txt")
            print("💾 Cache de embeddings configurado")
            
            # Menu principal
            while True:
                print("\n🎯 MENU PRINCIPAL SEMÂNTICO:")
                print("1. 💬 Chat semântico interativo")
                print("2. 🔬 Testes comparativos (Semântico vs Jaccard)")
                print("3. 📊 Estatísticas do sistema")
                print("4. 📁 Informações do arquivo e cache")
                print("5. 🧹 Limpar cache de embeddings")
                print("6. 🆘 Ajuda e exemplos")
                print("7. 🚪 Sair")
                
                try:
                    opcao = input("\nEscolha uma opção (1-7): ").strip()
                    
                    if opcao == '1':
                        chatbot.interface_chat()
                    elif opcao == '2':
                        chatbot.executar_testes_comparativos()
                    elif opcao == '3':
                        chatbot.mostrar_estatisticas()
                    elif opcao == '4':
                        chatbot.verificar_arquivo_info()
                    elif opcao == '5':
                        chatbot._limpar_cache()
                    elif opcao == '6':
                        chatbot._mostrar_ajuda_semantica()
                    elif opcao == '7':
                        print("👋 Encerrando sistema semântico...")
                        break
                    else:
                        print("❌ Opção inválida. Digite um número de 1 a 7.")
                        
                except KeyboardInterrupt:
                    print("\n👋 Encerrando...")
                    break
                except Exception as e:
                    print(f"❌ Erro no menu: {e}")
                    print("Tente novamente ou digite 7 para sair.")
        else:
            print("❌ Falha na inicialização do sistema")
            
    except ImportError as e:
        print(f"\n❌ ERRO DE DEPENDÊNCIAS:")
        print(f"   {e}")
        print(f"\n📦 INSTALE AS DEPENDÊNCIAS:")
        print(f"   pip install sentence-transformers scikit-learn")
        print(f"\n💡 OU execute a versão anterior sem embeddings semânticos")
        
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("  pip install sentence-transformers scikit-learn numpy")

if __name__ == "__main__":
    main()