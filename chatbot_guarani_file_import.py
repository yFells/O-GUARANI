#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot "O Guarani" - Versão Ultra Robusta com Import de Arquivo
Carrega o texto de O Guarani a partir do arquivo guarani.txt
"""

import numpy as np
import re
import os
from datetime import datetime
from typing import List, Dict
import time

class GuaraniChatbotUltraRobusta:
    """
    Chatbot O Guarani - Versão que funciona 100% sem erros
    Carrega texto do arquivo guarani.txt
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani (Versão Ultra Robusta)")
        print("=" * 60)
        
        # Configurações otimizadas
        self.chunk_size = 150
        self.overlap = 0.3
        self.similarity_threshold = 0.15
        self.top_chunks = 3
        
        # Estruturas de dados
        self.conversation_history = []
        self.processing_log = []
        self.performance_metrics = []
        self.text_chunks = []
        self.chunk_sentences = []
        
        # Stop words expandidas
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
        
        self._log("Sistema inicializado com sucesso")
    
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
            
            if self.texto_guarani:
                words = len(self.texto_guarani.split())
                lines = len(self.texto_guarani.split('\n'))
                chars = len(self.texto_guarani)
                
                print(f"\n📊 CONTEÚDO CARREGADO:")
                print(f"   • Caracteres: {chars:,}")
                print(f"   • Palavras: {words:,}")
                print(f"   • Linhas: {lines:,}")
                print(f"   • Primeiros 150 chars: {self.texto_guarani[:150]}...")
            else:
                print(f"\n❌ Nenhum conteúdo carregado")
                
        except Exception as e:
            print(f"❌ Erro ao verificar arquivo: {e}")
    
    def recarregar_arquivo(self):
        """Recarrega o arquivo guarani.txt"""
        print(f"\n🔄 RECARREGANDO ARQUIVO...")
        
        try:
            novo_texto = self._carregar_texto_arquivo()
            
            if novo_texto and len(novo_texto) > 100:
                self.texto_guarani = novo_texto
                
                # Limpar dados processados para forçar reprocessamento
                self.text_chunks = []
                self.chunk_sentences = []
                
                print(f"✅ Arquivo recarregado com sucesso!")
                print(f"📏 Novo tamanho: {len(self.texto_guarani)} caracteres")
                print(f"⚠️ Execute novamente as fases de processamento")
                return True
            else:
                print(f"❌ Falha ao recarregar arquivo")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao recarregar: {e}")
            return False
    
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
    
    def fase2_criar_chunks(self):
        """Fase 2: Criação de chunks"""
        self._log("=== FASE 2: CRIAÇÃO DE CHUNKS ===")
        
        if not self.texto_guarani:
            self._log("❌ Texto não carregado!")
            return False
        
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
        
        # Estatísticas
        if chunks:
            chunk_sizes = [len(chunk.split()) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            self._log(f"Chunks criados: {len(chunks)}")
            self._log(f"Tamanho médio: {avg_size:.1f} palavras")
        
        return True
    
    def calcular_similaridade_jaccard_melhorada(self, pergunta: str, texto: str) -> float:
        """Similaridade Jaccard otimizada sem dependências"""
        try:
            # Preprocessamento seguro
            pergunta_words = set()
            texto_words = set()
            
            # Extrair palavras da pergunta
            for word in re.findall(r'\b\w+\b', pergunta.lower()):
                if len(word) > 2 and word not in self.stop_words:
                    pergunta_words.add(word)
            
            # Extrair palavras do texto
            for word in re.findall(r'\b\w+\b', texto.lower()):
                if len(word) > 2 and word not in self.stop_words:
                    texto_words.add(word)
            
            # Verificar se há palavras válidas
            if not pergunta_words or not texto_words:
                return 0.0
            
            # Cálculo Jaccard
            intersection = len(pergunta_words & texto_words)
            union = len(pergunta_words | texto_words)
            jaccard = intersection / union if union > 0 else 0.0
            
            # Bonus para matches importantes
            important_words = pergunta_words - {'quem', 'qual', 'onde', 'como', 'quando', 'sobre', 'fale', 'conte', 'descreva'}
            exact_matches = len(important_words & texto_words)
            bonus = min(exact_matches * 0.1, 0.3)
            
            # Bonus para palavras-chave específicas
            key_words = {'peri', 'cecília', 'ceci', 'antonio', 'mariz', 'alvaro', 'isabel', 'aimores', 'guarani', 'alencar'}
            key_matches = len((pergunta_words & key_words) & (texto_words & key_words))
            key_bonus = min(key_matches * 0.15, 0.2)
            
            final_similarity = min(jaccard + bonus + key_bonus, 1.0)
            
            return float(final_similarity)  # Garantir que retorna float
            
        except Exception as e:
            self._log(f"Erro no cálculo de similaridade: {e}")
            return 0.0
    
    def fase3_responder_pergunta(self, pergunta: str) -> str:
        """Fase 3: Resposta à pergunta (versão ultra robusta)"""
        start_time = time.time()
        self._log(f"=== CONSULTA: {pergunta} ===")
        
        if not self.text_chunks:
            return "❌ Sistema não processado. Execute as fases anteriores."
        
        try:
            # Calcular similaridades de forma segura
            similarities = []
            for chunk in self.text_chunks:
                sim = self.calcular_similaridade_jaccard_melhorada(pergunta, chunk)
                similarities.append(float(sim))  # Garantir que é float
            
            # Verificar se temos similaridades válidas
            if not similarities:
                return "❌ Erro no cálculo de similaridades."
            
            # Criar resultados de forma segura
            chunk_results = []
            for i, sim in enumerate(similarities):
                chunk_results.append({
                    'chunk_id': i,
                    'chunk': self.text_chunks[i],
                    'similarity': float(sim),  # Garantir float
                    'sentences': self.chunk_sentences[i] if i < len(self.chunk_sentences) else []
                })
            
            # Ordenar por similaridade
            chunk_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Estatísticas seguras
            max_sim = chunk_results[0]['similarity'] if chunk_results else 0.0
            mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
            
            self._log(f"Similaridade máxima: {max_sim:.3f}")
            self._log(f"Similaridade média: {mean_sim:.3f}")
            
            # Filtrar chunks relevantes de forma segura
            relevant_chunks = []
            for chunk in chunk_results:
                if chunk['similarity'] >= self.similarity_threshold:
                    relevant_chunks.append(chunk)
            
            self._log(f"Chunks relevantes: {len(relevant_chunks)}")
            
            # Gerar resposta
            if not relevant_chunks:
                response = self._resposta_nao_encontrada(pergunta, max_sim)
            else:
                response = self._gerar_resposta_segura(pergunta, relevant_chunks[:self.top_chunks])
            
            # Métricas
            processing_time = time.time() - start_time
            self.performance_metrics.append({
                'pergunta': pergunta,
                'tempo': processing_time,
                'max_similarity': max_sim,
                'chunks_relevantes': len(relevant_chunks)
            })
            
            # Histórico
            self.conversation_history.append({
                'pergunta': pergunta,
                'resposta': response,
                'similaridade_max': max_sim,
                'chunks_usados': len(relevant_chunks),
                'tempo_resposta': processing_time,
                'timestamp': datetime.now()
            })
            
            self._log(f"Resposta gerada em {processing_time:.3f}s")
            return response
            
        except Exception as e:
            error_msg = f"❌ Erro inesperado: {e}"
            self._log(error_msg)
            return error_msg
    
    def _resposta_nao_encontrada(self, pergunta: str, max_sim: float) -> str:
        """Resposta quando não encontra informações"""
        base_msg = "Não encontrei informações específicas sobre sua pergunta no texto de 'O Guarani'."
        
        if max_sim > 0.1:
            suggestion = "\n\n💡 Tente reformular usando termos mais específicos da obra."
        elif max_sim > 0.05:
            suggestion = "\n\n💡 Use nomes de personagens ou eventos específicos."
        else:
            suggestion = "\n\n💡 Sua pergunta pode estar fora do escopo da obra."
        
        examples = """
\n📝 Exemplos de perguntas eficazes:
• "Quem é Peri?" ou "Fale sobre Peri"
• "Quem é Cecília?" ou "Descreva Ceci"
• "Qual a relação entre Peri e Cecília?"
• "Quem são os aimorés?"
• "Onde se passa a história?"
• "Quem é Dom Antônio de Mariz?"
• "Fale sobre Álvaro"
• "Quem é Isabel?"
"""
        
        confidence = f"\n\n🔴 Confiança muito baixa (similaridade máxima: {max_sim:.3f})"
        
        return base_msg + suggestion + examples + confidence
    
    def _gerar_resposta_segura(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta de forma segura"""
        if not chunks:
            return self._resposta_nao_encontrada(pergunta, 0.0)
        
        try:
            best_chunk = chunks[0]
            
            # Tentar busca por sentença se disponível
            if best_chunk.get('sentences'):
                sentences = best_chunk['sentences']
                best_sentence = ""
                best_sentence_sim = 0.0
                
                for sentence in sentences:
                    sim = self.calcular_similaridade_jaccard_melhorada(pergunta, sentence)
                    if sim > best_sentence_sim:
                        best_sentence_sim = sim
                        best_sentence = sentence
                
                # Se encontrou uma boa sentença, usar ela
                if best_sentence_sim > 0.2 and best_sentence:
                    confidence = self._calcular_confianca(best_sentence_sim)
                    return f"Com base em 'O Guarani':\n\n{best_sentence}\n\n{confidence}"
            
            # Usar chunk completo
            if len(chunks) == 1:
                main_content = chunks[0]['chunk']
                intro = "Com base no texto de 'O Guarani':\n\n"
            else:
                # Combinar chunks (limitado para evitar texto muito longo)
                combined_chunks = []
                total_length = 0
                for chunk in chunks[:2]:  # Máximo 2 chunks
                    chunk_text = chunk['chunk']
                    if total_length + len(chunk_text) < 600:
                        combined_chunks.append(chunk_text)
                        total_length += len(chunk_text)
                    else:
                        break
                
                main_content = ". ".join(combined_chunks)
                intro = "Combinando informações de 'O Guarani':\n\n"
            
            # Truncar se muito longo
            if len(main_content) > 500:
                main_content = main_content[:500] + "..."
            
            confidence = self._calcular_confianca(best_chunk['similarity'])
            return intro + main_content + "\n\n" + confidence
            
        except Exception as e:
            self._log(f"Erro na geração de resposta: {e}")
            return f"❌ Erro ao gerar resposta: {e}"
    
    def _calcular_confianca(self, similarity: float) -> str:
        """Calcula indicador de confiança"""
        try:
            sim = float(similarity)
            if sim > 0.5:
                return "🟢 Confiança muito alta"
            elif sim > 0.35:
                return "🟢 Confiança alta"
            elif sim > 0.25:
                return "🟡 Confiança moderada"
            elif sim > 0.15:
                return "🟠 Confiança baixa - considere reformular"
            else:
                return "🔴 Confiança muito baixa"
        except:
            return "⚠️ Confiança indeterminada"
    
    def executar_sistema_completo(self):
        """Executa todas as fases"""
        try:
            self._log("🚀 EXECUTANDO SISTEMA COMPLETO")
            
            if not self.fase1_analisar_texto():
                raise Exception("Erro na Fase 1")
            
            if not self.fase2_criar_chunks():
                raise Exception("Erro na Fase 2")
            
            self._log("✅ Sistema pronto para consultas!")
            return True
            
        except Exception as e:
            self._log(f"❌ Erro na execução: {e}")
            return False
    
    def executar_testes_automaticos(self):
        """Testes automáticos ultra robustos"""
        perguntas_teste = [
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Quem é Dom Antônio de Mariz?",
            "Descreva Álvaro",
            "Qual a relação entre Peri e Cecília?",
            "Quem Isabel ama?",
            "Quem são os aimorés?",
            "Fale sobre Loredano",
            "Onde se passa a história?",
            "Quando foi publicado O Guarani?",
            "Quais são os temas da obra?",
            "Como é descrita a natureza?",
            "Como fazer um bolo?",  # Deve ser rejeitada
            "Qual a capital da França?"  # Deve ser rejeitada
        ]
        
        print(f"\n🧪 EXECUTANDO TESTES AUTOMÁTICOS ({len(perguntas_teste)} perguntas)")
        print("=" * 70)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n📋 Teste {i:2d}/{len(perguntas_teste)}: {pergunta}")
            
            try:
                resposta = self.fase3_responder_pergunta(pergunta)
                ultimo_historico = self.conversation_history[-1]
                qualidade = self._avaliar_qualidade(ultimo_historico['similaridade_max'])
                
                resultado = {
                    'pergunta': pergunta,
                    'tempo': ultimo_historico['tempo_resposta'],
                    'similaridade': ultimo_historico['similaridade_max'],
                    'qualidade': qualidade
                }
                resultados.append(resultado)
                
                print(f"   ⏱️  {ultimo_historico['tempo_resposta']:.3f}s | 📊 {ultimo_historico['similaridade_max']:.3f} | {qualidade}")
                
                # Mostrar início da resposta se relevante
                if ultimo_historico['similaridade_max'] > 0.1:
                    print(f"   💬 {resposta[:80]}...")
                
            except Exception as e:
                print(f"   ❌ ERRO: {e}")
                resultado = {
                    'pergunta': pergunta,
                    'tempo': 0.0,
                    'similaridade': 0.0,
                    'qualidade': "❌ Erro"
                }
                resultados.append(resultado)
        
        # Relatório final
        self._relatorio_testes_seguro(resultados)
        return resultados
    
    def _avaliar_qualidade(self, similaridade: float) -> str:
        """Avalia qualidade da resposta"""
        try:
            sim = float(similaridade)
            if sim > 0.4:
                return "🟢 Excelente"
            elif sim > 0.25:
                return "🟡 Boa"
            elif sim > 0.15:
                return "🟠 Regular"
            elif sim > 0.05:
                return "🔴 Ruim"
            else:
                return "❌ Irrelevante"
        except:
            return "⚠️ Indeterminada"
    
    def _relatorio_testes_seguro(self, resultados: List[Dict]):
        """Relatório seguro dos testes"""
        print(f"\n📋 RELATÓRIO DOS TESTES")
        print("=" * 50)
        
        try:
            # Calcular métricas de forma segura
            tempos = [float(r.get('tempo', 0)) for r in resultados if r.get('tempo') is not None]
            similaridades = [float(r.get('similaridade', 0)) for r in resultados if r.get('similaridade') is not None]
            qualidades = [r.get('qualidade', 'Indeterminada') for r in resultados]
            
            if tempos:
                tempo_medio = sum(tempos) / len(tempos)
                print(f"📊 MÉTRICAS:")
                print(f"   • Tempo médio: {tempo_medio:.3f}s")
            
            if similaridades:
                sim_media = sum(similaridades) / len(similaridades)
                print(f"   • Similaridade média: {sim_media:.3f}")
                print(f"   • Similaridade máxima: {max(similaridades):.3f}")
            
            # Contagem de qualidades
            contadores = {}
            for qual in qualidades:
                contadores[qual] = contadores.get(qual, 0) + 1
            
            total = len(resultados)
            print(f"\n🎯 QUALIDADE:")
            
            qualidade_ordem = ["🟢 Excelente", "🟡 Boa", "🟠 Regular", "🔴 Ruim", "❌ Irrelevante", "❌ Erro"]
            for qual in qualidade_ordem:
                count = contadores.get(qual, 0)
                if count > 0:
                    percent = (count / total * 100) if total > 0 else 0
                    print(f"   • {qual}: {count}/{total} ({percent:.1f}%)")
            
            # Análise geral
            sucessos = contadores.get("🟢 Excelente", 0) + contadores.get("🟡 Boa", 0)
            taxa_sucesso = (sucessos / total * 100) if total > 0 else 0
            
            print(f"\n📈 ANÁLISE GERAL:")
            print(f"   • Taxa de sucesso: {taxa_sucesso:.1f}%")
            
            if taxa_sucesso > 70:
                print("   ✅ Sistema funcionando muito bem!")
            elif taxa_sucesso > 50:
                print("   🟡 Sistema funcionando adequadamente")
            else:
                print("   ⚠️ Sistema pode ser melhorado")
                
        except Exception as e:
            print(f"❌ Erro no relatório: {e}")
    
    def mostrar_estatisticas(self):
        """Estatísticas do sistema"""
        print(f"\n📊 ESTATÍSTICAS DO SISTEMA")
        print("=" * 40)
        
        try:
            print(f"📝 Chunks: {len(self.text_chunks)}")
            print(f"🔧 Threshold: {self.similarity_threshold}")
            print(f"📏 Tamanho chunks: {self.chunk_size} palavras")
            print(f"🔄 Sobreposição: {self.overlap * 100}%")
            print(f"💬 Consultas: {len(self.conversation_history)}")
            print(f"🛠️ Método: Jaccard otimizado")
            print(f"🛡️ Stop words: {len(self.stop_words)}")
            
            if self.texto_guarani:
                words = len(self.texto_guarani.split())
                chars = len(self.texto_guarani)
                print(f"📄 Texto carregado: {chars:,} chars, {words:,} palavras")
            
            if self.performance_metrics:
                tempos = [float(m.get('tempo', 0)) for m in self.performance_metrics if m.get('tempo')]
                if tempos:
                    print(f"⏱️ Tempo médio: {sum(tempos)/len(tempos):.3f}s")
            
            if self.conversation_history:
                similarities = [float(c.get('similaridade_max', 0)) for c in self.conversation_history if c.get('similaridade_max')]
                if similarities:
                    print(f"📈 Similaridade média: {sum(similarities)/len(similarities):.3f}")
                    
        except Exception as e:
            print(f"❌ Erro nas estatísticas: {e}")
    
    def interface_chat(self):
        """Interface de chat ultra robusta"""
        print(f"\n🤖 CHATBOT O GUARANI - CHAT INTERATIVO")
        print("=" * 50)
        print("Comandos especiais:")
        print("  • 'sair' - Encerrar chat")
        print("  • 'stats' - Ver estatísticas")
        print("  • 'teste' - Executar testes")
        print("  • 'help' - Mostrar ajuda")
        print("  • 'arquivo' - Info sobre arquivo")
        print("  • 'reload' - Recarregar arquivo")
        print("=" * 50)
        
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
                elif pergunta.lower() in ['teste', 'testes', 'test']:
                    self.executar_testes_automaticos()
                    continue
                elif pergunta.lower() in ['help', 'ajuda', '?']:
                    self._mostrar_ajuda()
                    continue
                elif pergunta.lower() in ['arquivo', 'file', 'info']:
                    self.verificar_arquivo_info()
                    continue
                elif pergunta.lower() in ['reload', 'recarregar', 'refresh']:
                    if self.recarregar_arquivo():
                        print("⚠️ Execute 'stats' para ver nova informação ou reinicie o processamento")
                    continue
                
                # Processar pergunta normal
                try:
                    resposta = self.fase3_responder_pergunta(pergunta)
                    print(f"\n🤖 {resposta}")
                except Exception as e:
                    print(f"\n❌ Erro ao processar pergunta: {e}")
                    print("Tente reformular sua pergunta.")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                print("Digite 'sair' para encerrar ou continue tentando.")
    
    def _mostrar_ajuda(self):
        """Mostra ajuda detalhada"""
        help_text = """
🆘 AJUDA - CHATBOT O GUARANI

📁 ARQUIVO:
   • O sistema carrega o texto do arquivo 'guarani.txt'
   • Se não encontrar, cria um arquivo de exemplo
   • Use 'arquivo' para ver informações do arquivo
   • Use 'reload' para recarregar o arquivo

📝 PERGUNTAS QUE FUNCIONAM BEM:

🧑 Personagens:
   • "Quem é Peri?" / "Fale sobre Peri"
   • "Quem é Cecília?" / "Descreva Ceci"
   • "Quem é Dom Antônio de Mariz?"
   • "Fale sobre Álvaro"
   • "Quem é Isabel?"

💕 Relacionamentos:
   • "Qual a relação entre Peri e Cecília?"
   • "Quem Isabel ama?"
   • "Por que Peri é devotado à Ceci?"

🏰 Contexto:
   • "Onde se passa a história?"
   • "Quando foi publicado O Guarani?"
   • "Como é o castelo?"

⚔️ Conflitos:
   • "Quem são os aimorés?"
   • "Fale sobre Loredano"
   • "Quais são os perigos?"

🎭 Temas:
   • "Quais são os temas principais?"
   • "Como é descrita a natureza?"
   • "O que a obra representa?"

💡 DICAS:
   • Seja específico nos nomes
   • Use termos da obra
   • Reformule se não obtiver boa resposta
   • Perguntas diretas funcionam melhor

❌ EVITE:
   • Perguntas muito vagas
   • Temas fora da obra
   • Perguntas sobre outros livros

🔧 COMANDOS:
   • 'arquivo' - Ver info do arquivo guarani.txt
   • 'reload' - Recarregar arquivo se modificado
   • 'stats' - Ver estatísticas do sistema
   • 'teste' - Executar bateria de testes
        """
        print(help_text)

def main():
    """Função principal ultra robusta"""
    print("🎯 CHATBOT O GUARANI - VERSÃO COM IMPORT DE ARQUIVO")
    print("=" * 60)
    print("Esta versão carrega o texto do arquivo 'guarani.txt'")
    print("Se o arquivo não existir, um exemplo será criado automaticamente.")
    print()
    
    try:
        chatbot = GuaraniChatbotUltraRobusta()
        
        # Mostrar informações do arquivo carregado
        chatbot.verificar_arquivo_info()
        
        if chatbot.executar_sistema_completo():
            print("\n✅ Sistema inicializado com sucesso!")
            print("🛡️ Versão ultra robusta - à prova de erros!")
            print("📁 Texto carregado do arquivo guarani.txt")
            
            # Menu principal
            while True:
                print("\n🎯 MENU PRINCIPAL:")
                print("1. 💬 Chat interativo")
                print("2. 🧪 Testes automáticos")
                print("3. 📊 Estatísticas do sistema")
                print("4. 📁 Informações do arquivo")
                print("5. 🔄 Recarregar arquivo")
                print("6. 🆘 Ajuda e exemplos")
                print("7. 🚪 Sair")
                
                try:
                    opcao = input("\nEscolha uma opção (1-7): ").strip()
                    
                    if opcao == '1':
                        chatbot.interface_chat()
                    elif opcao == '2':
                        chatbot.executar_testes_automaticos()
                    elif opcao == '3':
                        chatbot.mostrar_estatisticas()
                    elif opcao == '4':
                        chatbot.verificar_arquivo_info()
                    elif opcao == '5':
                        if chatbot.recarregar_arquivo():
                            print("✅ Arquivo recarregado! Reprocessando...")
                            if chatbot.executar_sistema_completo():
                                print("✅ Sistema reprocessado com sucesso!")
                            else:
                                print("❌ Erro no reprocessamento")
                    elif opcao == '6':
                        chatbot._mostrar_ajuda()
                    elif opcao == '7':
                        print("👋 Encerrando sistema...")
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
            
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        print("Verifique se Python e NumPy estão instalados corretamente.")
        print("Certifique-se de que o arquivo 'guarani.txt' existe ou pode ser criado.")

if __name__ == "__main__":
    main()