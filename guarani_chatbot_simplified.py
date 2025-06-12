#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot "O Guarani" - Versão Simplificada e Robusta
Implementação das melhorias sem dependências problemáticas
"""

import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional
import time

# Tentar importar bibliotecas opcionais
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn não disponível. Usando similaridade simplificada.")
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️ NLTK não disponível. Usando processamento simplificado.")
    NLTK_AVAILABLE = False

class GuaraniChatbotSimplified:
    """
    Chatbot especializado em "O Guarani" - Versão simplificada e robusta
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani (Versão Simplificada)")
        print("=" * 60)
        
        # Configurações otimizadas baseadas nas melhorias
        self.chunk_size = 150
        self.overlap = 0.3
        self.similarity_threshold = 0.15
        self.top_chunks = 3
        self.sentence_level_search = True
        
        # Inicializar estruturas de dados
        self.conversation_history = []
        self.processing_log = []
        self.performance_metrics = []
        self.text_chunks = []
        self.chunk_sentences = []
        
        # Stop words em português (lista expandida)
        self.stop_words = {
            'a', 'o', 'e', 'de', 'da', 'do', 'em', 'um', 'uma', 'com', 'para',
            'por', 'que', 'se', 'na', 'no', 'ao', 'aos', 'as', 'os', 'mais',
            'mas', 'ou', 'ter', 'ser', 'estar', 'seu', 'sua', 'seus', 'suas',
            'foi', 'são', 'dos', 'das', 'pela', 'pelo', 'sobre', 'até', 'sem',
            'muito', 'bem', 'já', 'ainda', 'só', 'pode', 'tem', 'vai', 'vem',
            'ele', 'ela', 'eles', 'elas', 'isso', 'isto', 'aquilo', 'quando',
            'onde', 'como', 'porque', 'então', 'assim', 'aqui', 'ali', 'lá'
        }
        
        # Texto expandido de O Guarani para demonstração
        self.texto_guarani = """
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
        """
        
        # Inicializar componentes
        self._init_vectorizer()
        
        self._log("Sistema inicializado com sucesso")
    
    def _log(self, message: str):
        """Registra eventos com timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(f"📝 {log_entry}")
    
    def _init_vectorizer(self):
        """Inicializa sistema de vetorização"""
        if SKLEARN_AVAILABLE:
            try:
                self.vectorizer = TfidfVectorizer(
                    max_features=3000,
                    stop_words=list(self.stop_words),
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    lowercase=True
                )
                self.use_tfidf = True
                self._log("TF-IDF inicializado com sucesso")
            except Exception as e:
                self._log(f"Erro TF-IDF: {e}. Usando similaridade simples.")
                self.use_tfidf = False
        else:
            self.use_tfidf = False
            self._log("Usando similaridade Jaccard simplificada")
    
    def fase1_carregar_texto(self):
        """Fase 1: Análise e carregamento do texto"""
        self._log("=== FASE 1: ANÁLISE DO TEXTO ===")
        
        # Estatísticas básicas
        chars = len(self.texto_guarani)
        words = self.texto_guarani.split()
        
        # Análise de sentenças (com ou sem NLTK)
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(self.texto_guarani, language='portuguese')
            except:
                sentences = re.split(r'[.!?]+', self.texto_guarani)
                sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = re.split(r'[.!?]+', self.texto_guarani)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Vocabulário único
        word_tokens = re.findall(r'\\b\\w+\\b', self.texto_guarani.lower())
        unique_words = set(word_tokens)
        content_words = unique_words - self.stop_words
        
        # Log de estatísticas
        self._log(f"Caracteres: {chars}")
        self._log(f"Palavras: {len(words)}")
        self._log(f"Sentenças: {len(sentences)}")
        self._log(f"Vocabulário único: {len(unique_words)}")
        self._log(f"Palavras de conteúdo: {len(content_words)}")
        
        return True
    
    def fase2_criar_chunks(self):
        """Fase 2: Criação de chunks otimizados"""
        self._log("=== FASE 2: CRIAÇÃO DE CHUNKS ===")
        
        # Limpeza do texto
        text = re.sub(r'\\n+', ' ', self.texto_guarani)
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Divisão em sentenças
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text, language='portuguese')
            except:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Criação de chunks com sobreposição
        chunks = []
        chunk_sentences_map = []
        current_chunk_sentences = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            if current_word_count + sentence_word_count <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
            else:
                # Finalizar chunk atual
                if current_chunk_sentences:
                    chunk_text = '. '.join(current_chunk_sentences) + '.'
                    chunks.append(chunk_text)
                    chunk_sentences_map.append(current_chunk_sentences.copy())
                
                # Sobreposição
                overlap_size = int(len(current_chunk_sentences) * self.overlap)
                if overlap_size > 0 and len(current_chunk_sentences) > overlap_size:
                    current_chunk_sentences = current_chunk_sentences[-overlap_size:]
                    current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_word_count = 0
                
                current_chunk_sentences.append(sentence)
                current_word_count += sentence_word_count
        
        # Último chunk
        if current_chunk_sentences:
            chunk_text = '. '.join(current_chunk_sentences) + '.'
            chunks.append(chunk_text)
            chunk_sentences_map.append(current_chunk_sentences.copy())
        
        self.text_chunks = chunks
        self.chunk_sentences = chunk_sentences_map
        
        # Estatísticas
        chunk_sizes = [len(chunk.split()) for chunk in chunks]
        self._log(f"Chunks criados: {len(chunks)}")
        self._log(f"Tamanho médio: {np.mean(chunk_sizes):.1f} palavras")
        
        return True
    
    def fase3_indexar(self):
        """Fase 3: Indexação dos chunks"""
        self._log("=== FASE 3: INDEXAÇÃO ===")
        
        if self.use_tfidf and SKLEARN_AVAILABLE:
            try:
                # Preprocessar chunks para TF-IDF
                processed_chunks = [self._preprocess_for_tfidf(chunk) for chunk in self.text_chunks]
                self.chunk_vectors = self.vectorizer.fit_transform(processed_chunks)
                self._log(f"Vetores TF-IDF criados: {self.chunk_vectors.shape}")
                return True
            except Exception as e:
                self._log(f"Erro na vetorização TF-IDF: {e}")
                self.use_tfidf = False
        
        # Fallback: usar similaridade simples
        self._log("Usando índice simplificado (sem vetorização)")
        return True
    
    def _preprocess_for_tfidf(self, text: str) -> str:
        """Pré-processamento para TF-IDF"""
        text = text.lower()
        words = re.findall(r'\\b\\w+\\b', text)
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def calcular_similaridade_simples(self, pergunta: str, texto: str) -> float:
        """Similaridade Jaccard melhorada"""
        # Preprocessamento
        pergunta_words = set(re.findall(r'\\b\\w+\\b', pergunta.lower()))
        pergunta_clean = pergunta_words - self.stop_words
        
        texto_words = set(re.findall(r'\\b\\w+\\b', texto.lower()))
        texto_clean = texto_words - self.stop_words
        
        # Similaridade Jaccard
        if not pergunta_clean or not texto_clean:
            return 0.0
        
        intersection = len(pergunta_clean & texto_clean)
        union = len(pergunta_clean | texto_clean)
        jaccard = intersection / union if union > 0 else 0
        
        # Bonus para matches de palavras importantes
        important_words = pergunta_clean - {'quem', 'qual', 'onde', 'como', 'quando', 'sobre', 'fale'}
        exact_matches = len(important_words & texto_clean)
        bonus = min(exact_matches * 0.1, 0.3)
        
        return min(jaccard + bonus, 1.0)
    
    def fase4_responder(self, pergunta: str) -> str:
        """Fase 4: Geração de resposta"""
        start_time = time.time()
        self._log(f"=== CONSULTA: {pergunta} ===")
        
        if not self.text_chunks:
            return "❌ Sistema não processado. Execute as fases anteriores."
        
        # Calcular similaridades
        if self.use_tfidf and hasattr(self, 'chunk_vectors'):
            try:
                # Usar TF-IDF
                processed_question = self._preprocess_for_tfidf(pergunta)
                question_vector = self.vectorizer.transform([processed_question])
                similarities = cosine_similarity(question_vector, self.chunk_vectors).flatten()
            except Exception as e:
                self._log(f"Erro TF-IDF, usando similaridade simples: {e}")
                similarities = [self.calcular_similaridade_simples(pergunta, chunk) for chunk in self.text_chunks]
        else:
            # Usar similaridade simples
            similarities = [self.calcular_similaridade_simples(pergunta, chunk) for chunk in self.text_chunks]
        
        # Processar resultados
        chunk_results = []
        for i, similarity in enumerate(similarities):
            chunk_results.append({
                'chunk_id': i,
                'chunk': self.text_chunks[i],
                'similarity': similarity,
                'sentences': self.chunk_sentences[i] if i < len(self.chunk_sentences) else []
            })
        
        # Ordenar por similaridade
        chunk_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        max_sim = chunk_results[0]['similarity'] if chunk_results else 0
        mean_sim = np.mean(similarities) if similarities else 0
        
        self._log(f"Similaridade máxima: {max_sim:.3f}")
        self._log(f"Similaridade média: {mean_sim:.3f}")
        
        # Filtrar chunks relevantes
        relevant_chunks = [chunk for chunk in chunk_results if chunk['similarity'] >= self.similarity_threshold]
        
        self._log(f"Chunks relevantes: {len(relevant_chunks)}")
        
        # Gerar resposta
        if not relevant_chunks:
            response = self._resposta_nao_encontrada(pergunta, max_sim)
        else:
            response = self._gerar_resposta(pergunta, relevant_chunks[:self.top_chunks])
        
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
    
    def _resposta_nao_encontrada(self, pergunta: str, max_sim: float) -> str:
        """Resposta quando não encontra informações relevantes"""
        base_msg = "Não encontrei informações específicas sobre sua pergunta no texto de 'O Guarani'."
        
        if max_sim > 0.1:
            suggestion = "\\n\\n💡 Tente reformular usando termos mais específicos da obra."
        elif max_sim > 0.05:
            suggestion = "\\n\\n💡 Use nomes de personagens ou eventos específicos."
        else:
            suggestion = "\\n\\n💡 Sua pergunta pode estar fora do escopo da obra."
        
        examples = """
\\n📝 Exemplos de perguntas eficazes:
• "Quem é Peri?" ou "Fale sobre Peri"
• "Quem é Cecília?" ou "Descreva Ceci"  
• "Qual a relação entre Peri e Cecília?"
• "Quem são os aimorés?"
• "Onde se passa a história?"
• "Quem é Dom Antônio de Mariz?"
"""
        
        confidence = f"\\n\\n🔴 Confiança muito baixa (similaridade: {max_sim:.3f})"
        
        return base_msg + suggestion + examples + confidence
    
    def _gerar_resposta(self, pergunta: str, chunks: List[Dict]) -> str:
        """Gera resposta baseada nos chunks relevantes"""
        if not chunks:
            return self._resposta_nao_encontrada(pergunta, 0)
        
        best_chunk = chunks[0]
        
        # Busca por sentença se disponível
        if self.sentence_level_search and best_chunk.get('sentences'):
            sentences = best_chunk['sentences']
            best_sentence = max(sentences, 
                              key=lambda s: self.calcular_similaridade_simples(pergunta, s))
            
            sentence_sim = self.calcular_similaridade_simples(pergunta, best_sentence)
            
            if sentence_sim > 0.2:
                # Usar sentença específica
                confidence = self._calcular_confianca(sentence_sim)
                return f"Com base em 'O Guarani':\\n\\n{best_sentence}\\n\\n{confidence}"
        
        # Usar chunk completo
        if len(chunks) == 1:
            main_content = chunks[0]['chunk']
            intro = "Com base no texto de 'O Guarani':\\n\\n"
        else:
            combined_content = ". ".join([chunk['chunk'] for chunk in chunks[:2]])
            main_content = combined_content
            intro = "Combinando informações de 'O Guarani':\\n\\n"
        
        # Truncar se muito longo
        if len(main_content) > 500:
            main_content = main_content[:500] + "..."
        
        confidence = self._calcular_confianca(best_chunk['similarity'])
        return intro + main_content + "\\n\\n" + confidence
    
    def _calcular_confianca(self, similarity: float) -> str:
        """Calcula indicador de confiança"""
        if similarity > 0.5:
            return "🟢 Confiança muito alta"
        elif similarity > 0.35:
            return "🟢 Confiança alta"
        elif similarity > 0.25:
            return "🟡 Confiança moderada"
        elif similarity > 0.15:
            return "🟠 Confiança baixa - considere reformular"
        else:
            return "🔴 Confiança muito baixa"
    
    def executar_sistema_completo(self):
        """Executa todas as fases do sistema"""
        try:
            self._log("🚀 EXECUTANDO SISTEMA COMPLETO")
            
            if not self.fase1_carregar_texto():
                raise Exception("Erro na Fase 1")
            
            if not self.fase2_criar_chunks():
                raise Exception("Erro na Fase 2")
            
            if not self.fase3_indexar():
                raise Exception("Erro na Fase 3")
            
            self._log("✅ Sistema pronto para consultas!")
            return True
            
        except Exception as e:
            self._log(f"❌ Erro na execução: {e}")
            return False
    
    def executar_testes_automaticos(self):
        """Executa testes automáticos"""
        perguntas_teste = [
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Quem é Dom Antônio de Mariz?",
            "Qual a relação entre Peri e Cecília?",
            "Quem são os aimorés?",
            "Onde se passa a história?",
            "Quando foi publicado O Guarani?",
            "Quais são os temas da obra?",
            "Como fazer um bolo?",  # Deve ser rejeitada
            "Qual a capital da França?"  # Deve ser rejeitada
        ]
        
        print(f"\\n🧪 EXECUTANDO TESTES AUTOMÁTICOS ({len(perguntas_teste)} perguntas)")
        print("=" * 70)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\\n📋 Teste {i:2d}/{len(perguntas_teste)}: {pergunta}")
            
            resposta = self.fase4_responder(pergunta)
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
            
            if ultimo_historico['similaridade_max'] > 0.1:
                print(f"   💬 {resposta[:80]}...")
        
        self._relatorio_testes(resultados)
        return resultados
    
    def _avaliar_qualidade(self, similaridade: float) -> str:
        """Avalia qualidade da resposta"""
        if similaridade > 0.35:
            return "🟢 Excelente"
        elif similaridade > 0.25:
            return "🟡 Boa"
        elif similaridade > 0.15:
            return "🟠 Regular"
        elif similaridade > 0.05:
            return "🔴 Ruim"
        else:
            return "❌ Irrelevante"
    
    def _relatorio_testes(self, resultados: List[Dict]):
        """Relatório dos testes"""
        print(f"\\n📋 RELATÓRIO DOS TESTES")
        print("=" * 50)
        
        tempos = [r['tempo'] for r in resultados]
        similaridades = [r['similaridade'] for r in resultados]
        qualidades = [r['qualidade'] for r in resultados]
        
        print(f"📊 MÉTRICAS:")
        print(f"   • Tempo médio: {np.mean(tempos):.3f}s")
        print(f"   • Similaridade média: {np.mean(similaridades):.3f}")
        
        excelentes = qualidades.count("🟢 Excelente")
        boas = qualidades.count("🟡 Boa")
        regulares = qualidades.count("🟠 Regular")
        ruins = qualidades.count("🔴 Ruim")
        irrelevantes = qualidades.count("❌ Irrelevante")
        
        total = len(resultados)
        print(f"\\n🎯 QUALIDADE:")
        print(f"   • Excelentes: {excelentes}/{total} ({excelentes/total*100:.1f}%)")
        print(f"   • Boas: {boas}/{total} ({boas/total*100:.1f}%)")
        print(f"   • Regulares: {regulares}/{total} ({regulares/total*100:.1f}%)")
        print(f"   • Ruins: {ruins}/{total} ({ruins/total*100:.1f}%)")
        print(f"   • Irrelevantes: {irrelevantes}/{total} ({irrelevantes/total*100:.1f}%)")
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas do sistema"""
        print(f"\\n📊 ESTATÍSTICAS DO SISTEMA")
        print("=" * 40)
        print(f"📝 Chunks: {len(self.text_chunks)}")
        print(f"🔧 Threshold: {self.similarity_threshold}")
        print(f"📏 Tamanho chunks: {self.chunk_size} palavras")
        print(f"🔄 Sobreposição: {self.overlap * 100}%")
        print(f"💬 Consultas: {len(self.conversation_history)}")
        print(f"🛠️ Método: {'TF-IDF' if self.use_tfidf else 'Jaccard'}")
        
        if self.performance_metrics:
            tempos = [m['tempo'] for m in self.performance_metrics]
            print(f"⏱️ Tempo médio: {np.mean(tempos):.3f}s")
    
    def interface_chat(self):
        """Interface de chat interativa"""
        print(f"\\n🤖 CHATBOT O GUARANI - CHAT INTERATIVO")
        print("=" * 50)
        print("Comandos: 'sair', 'stats', 'teste'")
        print("=" * 50)
        
        while True:
            try:
                pergunta = input("\\n💬 Sua pergunta: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                elif pergunta.lower() in ['stats', 'estatisticas']:
                    self.mostrar_estatisticas()
                    continue
                elif pergunta.lower() in ['teste', 'testes']:
                    self.executar_testes_automaticos()
                    continue
                
                if not pergunta:
                    continue
                
                resposta = self.fase4_responder(pergunta)
                print(f"\\n🤖 {resposta}")
                
            except KeyboardInterrupt:
                print("\\n👋 Encerrando...")
                break

def main():
    """Função principal"""
    print("🎯 CHATBOT O GUARANI - VERSÃO SIMPLIFICADA E ROBUSTA")
    print("=" * 60)
    
    chatbot = GuaraniChatbotSimplified()
    
    if chatbot.executar_sistema_completo():
        print("\\n✅ Sistema inicializado com sucesso!")
        
        # Menu
        while True:
            print("\\n🎯 MENU:")
            print("1. 💬 Chat interativo")
            print("2. 🧪 Testes automáticos")
            print("3. 📊 Estatísticas")
            print("4. 🚪 Sair")
            
            try:
                opcao = input("\\nEscolha (1-4): ").strip()
                
                if opcao == '1':
                    chatbot.interface_chat()
                elif opcao == '2':
                    chatbot.executar_testes_automaticos()
                elif opcao == '3':
                    chatbot.mostrar_estatisticas()
                elif opcao == '4':
                    print("👋 Encerrando...")
                    break
                else:
                    print("❌ Opção inválida.")
                    
            except KeyboardInterrupt:
                print("\\n👋 Encerrando...")
                break
    else:
        print("❌ Falha na inicialização")

if __name__ == "__main__":
    main()