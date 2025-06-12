#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot "O Guarani" - Demonstração Prática das Melhorias Implementadas
Versão executável para teste e validação
"""

import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Tuple

class GuaraniChatbotDemo:
    """
    Demonstração prática do Chatbot O Guarani com todas as melhorias implementadas
    """
    
    def __init__(self):
        print("🚀 Inicializando Chatbot O Guarani (Versão Melhorada)")
        print("=" * 60)
        
        # Configurações otimizadas baseadas nas sugestões
        self.chunk_size = 150      # ✅ Reduzido de 250 para 150 palavras
        self.overlap = 0.3         # ✅ Ajustado de 0.5 para 0.3
        self.similarity_threshold = 0.15  # ✅ Aumentado de 0.05 para 0.15
        self.top_chunks = 3
        self.sentence_level_search = True  # ✅ Nova funcionalidade
        
        # Stop words em português (reintroduzidas)
        self.stop_words = {
            'a', 'o', 'e', 'de', 'da', 'do', 'em', 'um', 'uma', 'com', 'para',
            'por', 'que', 'se', 'na', 'no', 'ao', 'aos', 'as', 'os', 'mais',
            'mas', 'ou', 'ter', 'ser', 'estar', 'seu', 'sua', 'seus', 'suas',
            'foi', 'são', 'dos', 'das', 'pela', 'pelo', 'sobre', 'até', 'sem',
            'muito', 'bem', 'já', 'ainda', 'só', 'pode', 'tem', 'vai', 'vem'
        }
        
        # Texto expandido para demonstração
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
        
        # Estruturas de dados
        self.text_chunks = []
        self.chunk_sentences = []  # Nova: sentenças por chunk
        self.conversation_history = []
        self.processing_log = []
        self.performance_metrics = []
        
        print("✅ Configuração inicial concluída")
    
    def log_evento(self, message: str):
        """Registra eventos com timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(f"📝 {log_entry}")
    
    def fase1_analise_texto(self):
        """
        Fase 1: Análise detalhada do texto
        Melhoria: Análise estatística mais completa
        """
        self.log_evento("=== FASE 1: ANÁLISE AVANÇADA DO TEXTO ===")
        
        # Estatísticas básicas
        chars = len(self.texto_guarani)
        words = self.texto_guarani.split()
        
        # Análise de sentenças
        sentences = re.split(r'[.!?]+', self.texto_guarani)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Análise de vocabulário
        word_tokens = re.findall(r'\b\w+\b', self.texto_guarani.lower())
        unique_words = set(word_tokens)
        content_words = unique_words - self.stop_words
        
        # Relatório detalhado
        stats = {
            "Caracteres totais": chars,
            "Palavras totais": len(words),
            "Sentenças": len(sentences),
            "Vocabulário único": len(unique_words),
            "Palavras de conteúdo": len(content_words),
            "Densidade lexical": f"{len(content_words)/len(unique_words)*100:.1f}%",
            "Média palavras/sentença": f"{len(words)/len(sentences):.1f}"
        }
        
        for key, value in stats.items():
            self.log_evento(f"{key}: {value}")
        
        return True
    
    def fase2_chunking_otimizado(self):
        """
        Fase 2: Criação de chunks otimizada
        Melhorias: Tamanho reduzido, melhor sobreposição, mapeamento de sentenças
        """
        self.log_evento("=== FASE 2: CHUNKING OTIMIZADO ===")
        
        # Limpeza do texto preservando estrutura
        text = re.sub(r'\n+', ' ', self.texto_guarani)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Segmentação em sentenças
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Criação de chunks com sobreposição otimizada
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
                
                # Calcular sobreposição
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
        
        # Estatísticas dos chunks
        chunk_sizes = [len(chunk.split()) for chunk in chunks]
        self.log_evento(f"Chunks criados: {len(chunks)}")
        self.log_evento(f"Tamanho médio: {np.mean(chunk_sizes):.1f} palavras")
        self.log_evento(f"Tamanho mínimo: {min(chunk_sizes)} palavras")
        self.log_evento(f"Tamanho máximo: {max(chunk_sizes)} palavras")
        
        return True
    
    def calcular_similaridade_melhorada(self, pergunta: str, texto: str) -> float:
        """
        Cálculo de similaridade melhorado
        Melhorias: Remoção de stop words, bonus para matches importantes
        """
        # Preprocessamento da pergunta
        pergunta_words = set(re.findall(r'\b\w+\b', pergunta.lower()))
        pergunta_content = pergunta_words - self.stop_words
        
        # Preprocessamento do texto
        texto_words = set(re.findall(r'\b\w+\b', texto.lower()))
        texto_content = texto_words - self.stop_words
        
        # Similaridade Jaccard básica
        intersection = len(pergunta_content & texto_content)
        union = len(pergunta_content | texto_content)
        jaccard_sim = intersection / union if union > 0 else 0
        
        # Bonus para palavras-chave importantes
        important_words = pergunta_content - {'quem', 'qual', 'onde', 'como', 'quando', 'sobre', 'fale', 'conte'}
        exact_matches = len(important_words & texto_content)
        bonus = min(exact_matches * 0.1, 0.3)  # Máximo 30% de bonus
        
        # Penalty para textos muito curtos
        min_content_size = min(len(pergunta_content), len(texto_content))
        if min_content_size < 3:
            penalty = 0.2
        else:
            penalty = 0
        
        final_similarity = max(0, min(1.0, jaccard_sim + bonus - penalty))
        return final_similarity
    
    def fase3_busca_nivel_sentenca(self, pergunta: str, chunk_id: int) -> Dict:
        """
        Fase 3: Busca refinada no nível de sentença
        Melhoria: Localizar a sentença mais relevante dentro do chunk
        """
        chunk = self.text_chunks[chunk_id]
        sentences = self.chunk_sentences[chunk_id]
        
        # Calcular similaridade para cada sentença
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            similarity = self.calcular_similaridade_melhorada(pergunta, sentence)
            sentence_scores.append({
                'sentence_id': i,
                'sentence': sentence,
                'similarity': similarity
            })
        
        # Encontrar melhor sentença
        best_sentence = max(sentence_scores, key=lambda x: x['similarity'])
        chunk_similarity = self.calcular_similaridade_melhorada(pergunta, chunk)
        
        return {
            'chunk_id': chunk_id,
            'chunk': chunk,
            'chunk_similarity': chunk_similarity,
            'best_sentence': best_sentence,
            'all_sentences': sentence_scores
        }
    
    def fase4_resposta_inteligente(self, pergunta: str) -> str:
        """
        Fase 4: Geração de resposta inteligente
        Melhorias: Threshold mais alto, busca por sentença, confiança melhorada
        """
        start_time = datetime.now()
        self.log_evento(f"=== CONSULTA: {pergunta} ===")
        
        if not self.text_chunks:
            return "❌ Sistema não processado. Execute as fases anteriores."
        
        # Calcular scores para todos os chunks
        chunk_results = []
        for i, chunk in enumerate(self.text_chunks):
            if self.sentence_level_search:
                result = self.fase3_busca_nivel_sentenca(pergunta, i)
            else:
                similarity = self.calcular_similaridade_melhorada(pergunta, chunk)
                result = {
                    'chunk_id': i,
                    'chunk': chunk,
                    'chunk_similarity': similarity,
                    'best_sentence': None
                }
            chunk_results.append(result)
        
        # Ordenar por similaridade
        chunk_results.sort(key=lambda x: x['chunk_similarity'], reverse=True)
        
        # Estatísticas
        similarities = [r['chunk_similarity'] for r in chunk_results]
        max_sim = max(similarities) if similarities else 0
        mean_sim = np.mean(similarities) if similarities else 0
        
        self.log_evento(f"Similaridade máxima: {max_sim:.3f}")
        self.log_evento(f"Similaridade média: {mean_sim:.3f}")
        
        # Filtrar resultados relevantes com threshold mais alto
        relevant_results = [
            result for result in chunk_results 
            if result['chunk_similarity'] >= self.similarity_threshold
        ]
        
        self.log_evento(f"Chunks relevantes encontrados: {len(relevant_results)}")
        
        # Gerar resposta
        if not relevant_results:
            response = self._resposta_nao_encontrada(pergunta, max_sim)
        else:
            response = self._gerar_resposta_otimizada(pergunta, relevant_results[:self.top_chunks])
        
        # Métricas de performance
        processing_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics.append({
            'pergunta': pergunta,
            'tempo_processamento': processing_time,
            'max_similarity': max_sim,
            'chunks_relevantes': len(relevant_results),
            'timestamp': datetime.now()
        })
        
        # Histórico expandido
        self.conversation_history.append({
            'pergunta': pergunta,
            'resposta': response,
            'similaridade_max': max_sim,
            'chunks_usados': len(relevant_results),
            'tempo_resposta': processing_time,
            'timestamp': datetime.now()
        })
        
        self.log_evento(f"Resposta gerada em {processing_time:.3f}s")
        return response
    
    def _resposta_nao_encontrada(self, pergunta: str, max_sim: float) -> str:
        """Resposta otimizada quando não encontra informações"""
        base_msg = "Não encontrei informações específicas sobre sua pergunta no texto de 'O Guarani'."
        
        if max_sim > 0.1:
            suggestion = "\n\n💡 Sugestão: Tente reformular usando termos mais próximos aos do texto original."
        elif max_sim > 0.05:
            suggestion = "\n\n💡 Sugestão: Use nomes específicos de personagens ou eventos da obra."
        else:
            suggestion = "\n\n💡 Sugestão: Sua pergunta pode estar fora do escopo da obra."
        
        examples = """
\n📝 Exemplos de perguntas eficazes:
• "Quem é Peri?" ou "Fale sobre Peri"
• "Quem é Cecília?" ou "Descreva Ceci"
• "Qual a relação entre Peri e Cecília?"
• "Quem são os aimorés?"
• "Onde se passa a história?"
• "Quem é Dom Antônio de Mariz?"
"""
        
        confidence = f"\n\n🔴 Confiança muito baixa (max. similaridade: {max_sim:.3f})"
        
        return base_msg + suggestion + examples + confidence
    
    def _gerar_resposta_otimizada(self, pergunta: str, results: List[Dict]) -> str:
        """Geração de resposta com busca por sentença"""
        if not results:
            return self._resposta_nao_encontrada(pergunta, 0)
        
        best_result = results[0]
        
        # Se temos busca por sentença e uma boa sentença foi encontrada
        if (self.sentence_level_search and 
            best_result.get('best_sentence') and 
            best_result['best_sentence']['similarity'] > 0.2):
            
            best_sentence = best_result['best_sentence']['sentence']
            sentence_sim = best_result['best_sentence']['similarity']
            
            # Adicionar contexto se necessário
            if len(results) > 1 and results[1]['chunk_similarity'] > 0.2:
                context_chunk = results[1]['chunk']
                if len(context_chunk) < 200:
                    additional_info = f"\n\nInformação adicional: {context_chunk}"
                else:
                    additional_info = f"\n\nInformação adicional: {context_chunk[:200]}..."
            else:
                additional_info = ""
            
            confidence = self._calcular_indicador_confianca(sentence_sim)
            
            response = f"Com base em 'O Guarani':\n\n{best_sentence}{additional_info}\n\n{confidence}"
            
        else:
            # Resposta baseada em chunk completo
            if len(results) == 1:
                main_content = results[0]['chunk']
                intro = "Com base no texto de 'O Guarani':\n\n"
            else:
                combined_content = ". ".join([r['chunk'] for r in results[:2]])
                main_content = combined_content
                intro = "Combinando informações de 'O Guarani':\n\n"
            
            # Truncar se muito longo
            if len(main_content) > 600:
                main_content = main_content[:600] + "..."
            
            confidence = self._calcular_indicador_confianca(best_result['chunk_similarity'])
            response = intro + main_content + "\n\n" + confidence
        
        return response
    
    def _calcular_indicador_confianca(self, similarity: float) -> str:
        """Indicadores de confiança mais precisos"""
        if similarity > 0.6:
            return "🟢 Confiança muito alta"
        elif similarity > 0.4:
            return "🟢 Confiança alta"
        elif similarity > 0.25:
            return "🟡 Confiança moderada"
        elif similarity > 0.15:
            return "🟠 Confiança baixa - considere reformular"
        else:
            return "🔴 Confiança muito baixa"
    
    def mostrar_estatisticas_completas(self):
        """Estatísticas detalhadas do sistema"""
        print("\n📊 ESTATÍSTICAS COMPLETAS DO SISTEMA")
        print("=" * 60)
        
        # Configurações
        print("🔧 CONFIGURAÇÕES:")
        print(f"   • Tamanho dos chunks: {self.chunk_size} palavras")
        print(f"   • Sobreposição: {self.overlap * 100}%")
        print(f"   • Threshold de similaridade: {self.similarity_threshold}")
        print(f"   • Busca por sentença: {'Ativada' if self.sentence_level_search else 'Desativada'}")
        print(f"   • Stop words removidas: {len(self.stop_words)}")
        
        # Dados processados
        print(f"\n📚 DADOS PROCESSADOS:")
        print(f"   • Total de chunks: {len(self.text_chunks)}")
        if self.text_chunks:
            chunk_sizes = [len(chunk.split()) for chunk in self.text_chunks]
            print(f"   • Tamanho médio dos chunks: {np.mean(chunk_sizes):.1f} palavras")
            print(f"   • Menor chunk: {min(chunk_sizes)} palavras")
            print(f"   • Maior chunk: {max(chunk_sizes)} palavras")
        
        # Histórico de consultas
        print(f"\n💬 HISTÓRICO DE CONSULTAS:")
        print(f"   • Total de consultas: {len(self.conversation_history)}")
        
        if self.performance_metrics:
            tempos = [m['tempo_processamento'] for m in self.performance_metrics]
            similarities = [m['max_similarity'] for m in self.performance_metrics]
            
            print(f"   • Tempo médio de resposta: {np.mean(tempos):.3f}s")
            print(f"   • Tempo mínimo: {min(tempos):.3f}s")
            print(f"   • Tempo máximo: {max(tempos):.3f}s")
            print(f"   • Similaridade média: {np.mean(similarities):.3f}")
            print(f"   • Melhor similaridade: {max(similarities):.3f}")
        
        # Qualidade das respostas
        if self.conversation_history:
            print(f"\n🎯 QUALIDADE DAS RESPOSTAS:")
            alta_confianca = sum(1 for c in self.conversation_history if c['similaridade_max'] > 0.4)
            media_confianca = sum(1 for c in self.conversation_history if 0.2 <= c['similaridade_max'] <= 0.4)
            baixa_confianca = sum(1 for c in self.conversation_history if c['similaridade_max'] < 0.2)
            
            total = len(self.conversation_history)
            print(f"   • Alta confiança: {alta_confianca}/{total} ({alta_confianca/total*100:.1f}%)")
            print(f"   • Média confiança: {media_confianca}/{total} ({media_confianca/total*100:.1f}%)")
            print(f"   • Baixa confiança: {baixa_confianca}/{total} ({baixa_confianca/total*100:.1f}%)")
    
    def mostrar_historico_detalhado(self):
        """Histórico detalhado das conversas"""
        if not self.conversation_history:
            print("📭 Nenhuma conversa no histórico.")
            return
        
        print(f"\n📚 HISTÓRICO DETALHADO ({len(self.conversation_history)} conversas)")
        print("=" * 70)
        
        for i, conv in enumerate(self.conversation_history, 1):
            timestamp = conv['timestamp'].strftime("%H:%M:%S")
            print(f"\n{i}. [{timestamp}] {conv['pergunta']}")
            print(f"   ⏱️  Tempo: {conv['tempo_resposta']:.3f}s")
            print(f"   📊 Similaridade: {conv['similaridade_max']:.3f}")
            print(f"   📝 Chunks usados: {conv['chunks_usados']}")
            print(f"   💭 Resposta: {conv['resposta'][:100]}...")
    
    def executar_testes_abrangentes(self):
        """Testes automáticos abrangentes do sistema"""
        perguntas_teste = [
            # Personagens principais
            "Quem é Peri?",
            "Fale sobre Cecília",
            "Quem é Dom Antônio de Mariz?",
            "Descreva Álvaro",
            
            # Relacionamentos
            "Qual a relação entre Peri e Cecília?",
            "Quem Isabel ama?",
            
            # Antagonistas
            "Quem são os aimorés?",
            "Fale sobre Loredano",
            
            # Contexto e temas
            "Onde se passa a história?",
            "Quando foi publicado O Guarani?",
            "Quais são os temas da obra?",
            "Como é descrita a natureza?",
            
            # Perguntas mais específicas
            "Por que Dom Antônio veio ao Brasil?",
            "O que representa Peri na obra?",
            
            # Perguntas que devem ter baixa similaridade
            "Qual a receita do bolo de chocolate?",
            "Como funciona um computador?"
        ]
        
        print(f"\n🧪 EXECUTANDO TESTES ABRANGENTES ({len(perguntas_teste)} perguntas)")
        print("=" * 70)
        
        resultados = []
        
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n📋 Teste {i:2d}/{len(perguntas_teste)}: {pergunta}")
            
            start_time = datetime.now()
            resposta = self.fase4_resposta_inteligente(pergunta)
            tempo_total = (datetime.now() - start_time).total_seconds()
            
            # Analisar qualidade da resposta
            ultimo_historico = self.conversation_history[-1]
            qualidade = self._avaliar_qualidade_resposta(ultimo_historico['similaridade_max'])
            
            resultado = {
                'numero': i,
                'pergunta': pergunta,
                'tempo': tempo_total,
                'similaridade': ultimo_historico['similaridade_max'],
                'chunks_usados': ultimo_historico['chunks_usados'],
                'qualidade': qualidade
            }
            resultados.append(resultado)
            
            print(f"   ⏱️  {tempo_total:.3f}s | 📊 {ultimo_historico['similaridade_max']:.3f} | {qualidade}")
            
            # Mostrar início da resposta para perguntas relevantes
            if ultimo_historico['similaridade_max'] > 0.1:
                print(f"   💬 {resposta[:80]}...")
        
        # Relatório final dos testes
        self._gerar_relatorio_final_testes(resultados)
        
        return resultados
    
    def _avaliar_qualidade_resposta(self, similaridade: float) -> str:
        """Avalia qualidade baseada na similaridade"""
        if similaridade > 0.4:
            return "🟢 Excelente"
        elif similaridade > 0.25:
            return "🟡 Boa"
        elif similaridade > 0.15:
            return "🟠 Regular"
        elif similaridade > 0.05:
            return "🔴 Ruim"
        else:
            return "❌ Irrelevante"
    
    def _gerar_relatorio_final_testes(self, resultados: List[Dict]):
        """Relatório detalhado dos testes"""
        print(f"\n📋 RELATÓRIO FINAL DOS TESTES")
        print("=" * 70)
        
        # Métricas gerais
        tempos = [r['tempo'] for r in resultados]
        similaridades = [r['similaridade'] for r in resultados]
        
        print(f"📊 MÉTRICAS GERAIS:")
        print(f"   • Testes executados: {len(resultados)}")
        print(f"   • Tempo total: {sum(tempos):.2f}s")
        print(f"   • Tempo médio por pergunta: {np.mean(tempos):.3f}s")
        print(f"   • Tempo máximo: {max(tempos):.3f}s")
        print(f"   • Similaridade média: {np.mean(similaridades):.3f}")
        print(f"   • Similaridade máxima: {max(similaridades):.3f}")
        print(f"   • Similaridade mínima: {min(similaridades):.3f}")
        
        # Distribuição de qualidade
        qualidades = [r['qualidade'] for r in resultados]
        excelentes = qualidades.count("🟢 Excelente")
        boas = qualidades.count("🟡 Boa")
        regulares = qualidades.count("🟠 Regular")
        ruins = qualidades.count("🔴 Ruim")
        irrelevantes = qualidades.count("❌ Irrelevante")
        
        total = len(resultados)
        print(f"\n🎯 DISTRIBUIÇÃO DE QUALIDADE:")
        print(f"   • Excelentes: {excelentes:2d}/{total} ({excelentes/total*100:5.1f}%)")
        print(f"   • Boas:       {boas:2d}/{total} ({boas/total*100:5.1f}%)")
        print(f"   • Regulares:  {regulares:2d}/{total} ({regulares/total*100:5.1f}%)")
        print(f"   • Ruins:      {ruins:2d}/{total} ({ruins/total*100:5.1f}%)")
        print(f"   • Irrelevantes: {irrelevantes:2d}/{total} ({irrelevantes/total*100:5.1f}%)")
        
        # Melhores e piores resultados
        resultados_ordenados = sorted(resultados, key=lambda x: x['similaridade'], reverse=True)
        
        print(f"\n🏆 TOP 3 MELHORES RESULTADOS:")
        for i, resultado in enumerate(resultados_ordenados[:3], 1):
            print(f"   {i}. {resultado['pergunta'][:40]}... | {resultado['similaridade']:.3f}")
        
        print(f"\n⚠️ TOP 3 PIORES RESULTADOS:")
        for i, resultado in enumerate(resultados_ordenados[-3:], 1):
            print(f"   {i}. {resultado['pergunta'][:40]}... | {resultado['similaridade']:.3f}")
        
        # Recomendações
        taxa_sucesso = (excelentes + boas) / total
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if taxa_sucesso > 0.7:
            print("   ✅ Sistema funcionando bem! Parâmetros otimizados.")
        elif taxa_sucesso > 0.5:
            print("   🟡 Sistema com performance razoável. Considere ajustes nos parâmetros.")
        else:
            print("   ⚠️ Sistema precisa de melhorias:")
            print("      - Verificar qualidade do texto de entrada")
            print("      - Ajustar threshold de similaridade")
            print("      - Revisar algoritmo de chunking")
        
        if np.mean(similaridades) < 0.2:
            print("   📉 Similaridades baixas detectadas:")
            print("      - Verificar remoção de stop words")
            print("      - Considerar técnicas de normalização textual")
            print("      - Avaliar algoritmo de similaridade")
    
    def executar_sistema_completo_melhorado(self):
        """Execução completa do sistema com todas as melhorias"""
        print("🚀 EXECUTANDO SISTEMA COMPLETO COM MELHORIAS")
        print("=" * 70)
        
        try:
            # Fase 1: Análise do texto
            if not self.fase1_analise_texto():
                raise Exception("Falha na análise do texto")
            
            # Fase 2: Chunking otimizado
            if not self.fase2_chunking_otimizado():
                raise Exception("Falha no processamento dos chunks")
            
            print("\n✅ SISTEMA INICIALIZADO COM SUCESSO!")
            print("🎯 Todas as melhorias foram implementadas:")
            print("   ✅ Threshold de similaridade aumentado para 0.15")
            print("   ✅ Tamanho dos chunks reduzido para 150 palavras")
            print("   ✅ Stop words reintroduzidas no processamento")
            print("   ✅ Busca refinada no nível de sentenças")
            print("   ✅ Sistema de confiança melhorado")
            print("   ✅ Métricas de performance expandidas")
            
            return True
            
        except Exception as e:
            self.log_evento(f"❌ Erro na execução: {e}")
            return False
    
    def interface_interativa_melhorada(self):
        """Interface de usuário melhorada"""
        print("\n" + "="*70)
        print("🤖 CHATBOT O GUARANI - VERSÃO MELHORADA")
        print("Assistente especializado na obra de José de Alencar")
        print(f"Threshold: {self.similarity_threshold} | Chunks: {len(self.text_chunks)} | Busca: Nível de sentença")
        print("\n📋 Comandos disponíveis:")
        print("   💬 Digite sua pergunta normalmente")
        print("   📊 'stats' - Estatísticas completas")
        print("   📚 'historico' - Histórico detalhado")
        print("   🧪 'teste' - Executar testes automáticos")
        print("   ❓ 'ajuda' - Mostrar exemplos de perguntas")
        print("   🚪 'sair' - Encerrar o sistema")
        print("="*70)
        
        while True:
            try:
                pergunta = input("\n💬 Sua pergunta: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Até logo!")
                    break
                elif pergunta.lower() in ['stats', 'estatisticas', 'estatísticas']:
                    self.mostrar_estatisticas_completas()
                    continue
                elif pergunta.lower() in ['historico', 'histórico', 'history']:
                    self.mostrar_historico_detalhado()
                    continue
                elif pergunta.lower() in ['teste', 'testes', 'test']:
                    self.executar_testes_abrangentes()
                    continue
                elif pergunta.lower() in ['ajuda', 'help', 'exemplos']:
                    self._mostrar_ajuda_detalhada()
                    continue
                
                if not pergunta:
                    print("⚠️  Digite uma pergunta ou comando.")
                    continue
                
                # Processar pergunta
                resposta = self.fase4_resposta_inteligente(pergunta)
                print(f"\n🤖 {resposta}")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando...")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
    
    def _mostrar_ajuda_detalhada(self):
        """Ajuda detalhada com exemplos"""
        help_text = """
🆘 AJUDA DETALHADA - CHATBOT O GUARANI

📝 TIPOS DE PERGUNTAS QUE FUNCIONAM BEM:

🧑 Sobre personagens principais:
   • "Quem é Peri?" / "Descreva Peri"
   • "Fale sobre Cecília" / "Quem é Ceci?"
   • "Quem é Dom Antônio de Mariz?"
   • "Descreva Álvaro"

💕 Sobre relacionamentos:
   • "Qual a relação entre Peri e Cecília?"
   • "Quem Isabel ama?"
   • "Por que Peri é devotado à Ceci?"

🏰 Sobre contexto e cenário:
   • "Onde se passa a história?"
   • "Quando foi publicado O Guarani?"
   • "Como é o castelo de Dom Antônio?"

⚔️ Sobre conflitos e antagonistas:
   • "Quem são os aimorés?"
   • "Fale sobre Loredano"
   • "Quais são os perigos na história?"

🎭 Sobre temas e estilo:
   • "Quais são os temas principais?"
   • "Como é descrita a natureza?"
   • "O que a obra representa?"

💡 DICAS PARA MELHORES RESPOSTAS:
   • Use nomes específicos de personagens
   • Seja direto e claro na pergunta
   • Reformule se a resposta não for satisfatória
   • Perguntas sobre a obra têm melhor resultado que perguntas gerais

⚠️ EVITE:
   • Perguntas muito vagas ou genéricas
   • Temas fora do escopo da obra
   • Perguntas sobre outros livros ou autores
        """
        print(help_text)

def main():
    """Função principal para demonstração completa"""
    print("🎯 DEMONSTRAÇÃO COMPLETA DO CHATBOT O GUARANI MELHORADO")
    print("📚 Implementando todas as melhorias sugeridas no documento")
    print("=" * 80)
    
    # Criar instância do chatbot
    chatbot = GuaraniChatbotDemo()
    
    # Executar sistema completo
    if chatbot.executar_sistema_completo_melhorado():
        print("\n🎉 SISTEMA PRONTO PARA USO!")
        
        # Menu principal
        while True:
            print("\n🎯 MENU PRINCIPAL:")
            print("1. 💬 Iniciar chat interativo")
            print("2. 🧪 Executar testes automáticos")
            print("3. 📊 Ver estatísticas do sistema")
            print("4. 📚 Ver histórico (se houver)")
            print("5. ❓ Ver ajuda e exemplos")
            print("6. 🚪 Sair")
            
            try:
                opcao = input("\nEscolha uma opção (1-6): ").strip()
                
                if opcao == '1':
                    chatbot.interface_interativa_melhorada()
                elif opcao == '2':
                    chatbot.executar_testes_abrangentes()
                elif opcao == '3':
                    chatbot.mostrar_estatisticas_completas()
                elif opcao == '4':
                    chatbot.mostrar_historico_detalhado()
                elif opcao == '5':
                    chatbot._mostrar_ajuda_detalhada()
                elif opcao == '6':
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