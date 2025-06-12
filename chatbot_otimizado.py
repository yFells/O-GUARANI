#!/usr/bin/env python3
"""
Chatbot "O Guarani" - Versão Otimizada Final
Sistema corrigido que encontra chunks relevantes com eficiência
"""

import os
import numpy as np
import re
from typing import List, Dict, Tuple
from datetime import datetime

# Verificação e instalação de dependências
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("📦 Instalando dependências necessárias...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "numpy"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

class GuaraniChatbotOtimizado:
    """
    Chatbot otimizado para responder perguntas sobre "O Guarani"
    Versão corrigida que resolve problemas de detecção de chunks
    """
    
    def __init__(self, debug=True):
        self.debug = debug
        self._log("🚀 Inicializando Chatbot O Guarani (Versão Otimizada)")
        
        # Configurações otimizadas
        self.similarity_threshold = 0.05  # Limiar muito baixo
        self.min_adaptive_threshold = 0.01  # Limiar mínimo adaptativo
        self.top_chunks = 3
        self.chunk_overlap = 0.3  # Sobreposição entre chunks
        
        # Dados do sistema
        self.text_chunks = []
        self.chunk_vectors = None
        self.vectorizer = None
        self.original_text = ""
        
        # Histórico detalhado
        self.conversation_history = []
        self.processing_log = []
        
        # Executar inicialização
        if self._carregar_e_processar():
            self._log("✅ Sistema inicializado com sucesso!")
        else:
            self._log("❌ Erro na inicialização")
    
    def _log(self, message: str):
        """Registra eventos no histórico"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.debug:
            print(f"📝 {log_entry}")
    
    def _carregar_e_processar(self) -> bool:
        """Carrega texto e processa em chunks"""
        try:
            # Fase 1: Carregar texto
            texto = self._carregar_texto()
            if not texto:
                return False
            
            # Fase 2: Criar chunks otimizados
            self.text_chunks = self._criar_chunks_otimizados(texto)
            if not self.text_chunks:
                self._log("Erro: Nenhum chunk criado")
                return False
            
            # Fase 3: Vetorização otimizada
            return self._vetorizar_chunks()
            
        except Exception as e:
            self._log(f"Erro no processamento: {e}")
            return False
    
    def _carregar_texto(self) -> str:
        """Carrega texto do arquivo ou usa fallback"""
        # Tentar carregar guarani.txt
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                with open('guarani.txt', 'r', encoding=encoding) as arquivo:
                    texto = arquivo.read()
                    self._log(f"📖 Arquivo carregado com {encoding}: {len(texto):,} chars")
                    
                    if len(texto) < 1000:
                        self._log("⚠️  Arquivo pequeno, pode estar incompleto")
                    
                    self.original_text = texto
                    return texto
                    
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                self._log("📄 Arquivo guarani.txt não encontrado, usando texto de demonstração")
                break
        
        # Usar texto de fallback expandido
        self.original_text = self._texto_demonstracao()
        return self.original_text
    
    def _texto_demonstracao(self) -> str:
        """Texto de demonstração expandido para testes"""
        return """
        CAPÍTULO I - A CHEGADA DE PERI
        
        Dom Antônio de Mariz era um fidalgo português que se estabeleceu no Brasil em 1604.
        Ele construiu um castelo fortificado às margens do rio Paquequer, na região que hoje
        é conhecida como Petrópolis. Dom Antônio era um homem justo e honrado, respeitado 
        por todos os habitantes da região.
        
        Peri era um índio goitacá de força excepcional e lealdade inquebrantável. Quando
        apareceu pela primeira vez diante do castelo, Dom Antônio o recebeu com desconfiança,
        como era natural entre colonos portugueses e indígenas naquela época.
        
        CAPÍTULO II - A EVOLUÇÃO DA RELAÇÃO
        
        A relação entre Peri e Dom Antônio evoluiu gradualmente ao longo da história.
        No início, Dom Antônio mantinha distância do índio, vendo-o com os preconceitos
        típicos de sua época. Porém, Peri demonstrou sua lealdade salvando Cecília várias vezes.
        
        Com o passar do tempo, Dom Antônio passou a reconhecer a nobreza de caráter de Peri.
        O fidalgo português começou a confiar no índio e até mesmo a consultá-lo sobre
        assuntos importantes relacionados à defesa do castelo.
        
        CAPÍTULO III - CECÍLIA E O AMOR DE PERI
        
        Cecília era a filha querida de Dom Antônio de Mariz. Ela era conhecida por todos
        como Ceci. Cecília representava a pureza e inocência da jovem portuguesa no Brasil.
        Peri devotava amor incondicional a Cecília, adorando-a como uma divindade.
        
        A relação entre Peri e Cecília era platônica e idealizada. O índio via na jovem
        portuguesa a personificação de todas as virtudes. Cecília, por sua vez, desenvolvia
        afeição especial por Peri, reconhecendo sua pureza de alma.
        
        CAPÍTULO IV - OS DEMAIS PERSONAGENS
        
        Álvaro era primo de Cecília e habitava o castelo. Ele representava o jovem português
        civilizado e culto. Álvaro desenvolveu amizade e respeito por Peri, reconhecendo
        suas qualidades excepcionais e sua honra natural.
        
        Isabel era irmã de Cecília, uma jovem de temperamento impetuoso e apaixonado.
        Seus amores por Álvaro eram conhecidos de todos no castelo. Isabel representava
        a paixão ardente em contraste com a pureza de Cecília.
        
        CAPÍTULO V - OS INIMIGOS AIMORÉS
        
        Os aimorés eram uma tribo feroz que habitava as florestas circunvizinhas ao castelo.
        Eram inimigos naturais dos goitacás, tribo à qual pertencia Peri. Constantemente
        atacavam os habitantes da região, espalhando terror e destruição por onde passavam.
        
        Peri lutou bravamente contra os aimorés para proteger a família de Dom Antônio.
        Esta proteção incansável fortaleceu ainda mais os laços entre o índio e o fidalgo
        português, consolidando uma relação de confiança mútua.
        
        CAPÍTULO VI - A NATUREZA BRASILEIRA
        
        A natureza brasileira era exuberante e selvagem. José de Alencar descreveu
        detalhadamente as florestas imensas, os rios cristalinos e a fauna diversificada
        da região. A paisagem servia como cenário grandioso para as aventuras dos personagens.
        
        Peri conhecia todos os segredos da floresta. Caçava com destreza incomparável,
        seguia rastros como nenhum outro índio, e movia-se pela mata com a agilidade
        de um felino. Sua força era lendária entre os índios da região.
        
        CAPÍTULO VII - O CLÍMAX E A REDENÇÃO
        
        No final da história, Dom Antônio confiou completamente em Peri. Quando o castelo
        foi atacado pelos aimorés, Dom Antônio sabia que apenas Peri poderia salvar sua filha.
        Esta confiança total representa o ápice da evolução de sua relação.
        
        A evolução da relação entre Dom Antônio e Peri mostra como o respeito mútuo
        pode superar preconceitos raciais e culturais. O fidalgo português reconheceu
        que Peri tinha mais honra que muitos homens considerados civilizados.
        
        EPÍLOGO - O SIMBOLISMO DA OBRA
        
        O romance de José de Alencar simboliza o encontro entre duas culturas: a indígena
        e a europeia. A união final de Peri e Cecília representa a formação da raça brasileira,
        mestiça e forte, que herdaria as melhores qualidades de ambos os povos.
        """
    
    def _criar_chunks_otimizados(self, texto: str) -> List[str]:
        """Cria chunks otimizados para busca"""
        self._log("🔄 Criando chunks otimizados...")
        
        # Limpeza inicial
        texto_limpo = self._limpar_texto(texto)
        
        chunks = []
        
        # Estratégia 1: Dividir por capítulos se existirem
        if 'CAPÍTULO' in texto_limpo or 'CAPITULO' in texto_limpo:
            capitulos = re.split(r'CAP[ÍI]TULO\s+[IVX\d]+', texto_limpo)
            for cap in capitulos:
                if len(cap.strip()) > 100:
                    # Subdividir capítulos longos em parágrafos
                    paragrafos = [p.strip() for p in cap.split('\n\n') if len(p.strip()) > 50]
                    chunks.extend(paragrafos)
        else:
            # Estratégia 2: Dividir por parágrafos
            paragrafos = [p.strip() for p in texto_limpo.split('\n\n') if len(p.strip()) > 50]
            if paragrafos:
                chunks = paragrafos
            else:
                # Estratégia 3: Dividir por sentenças
                sentencas = re.split(r'[.!?]+', texto_limpo)
                chunks = [s.strip() for s in sentencas if len(s.strip()) > 30]
        
        # Criar chunks com sobreposição para capturar contexto
        chunks_com_overlap = self._adicionar_overlap(chunks)
        
        self._log(f"📝 Criados {len(chunks_com_overlap)} chunks (incluindo overlap)")
        
        # Debug: mostrar exemplos
        if self.debug and chunks_com_overlap:
            for i, chunk in enumerate(chunks_com_overlap[:3]):
                self._log(f"Exemplo chunk {i+1}: {chunk[:80]}...")
        
        return chunks_com_overlap
    
    def _adicionar_overlap(self, chunks: List[str]) -> List[str]:
        """Adiciona sobreposição entre chunks para manter contexto"""
        if len(chunks) <= 1:
            return chunks
        
        chunks_overlap = []
        
        for i, chunk in enumerate(chunks):
            chunks_overlap.append(chunk)
            
            # Adicionar chunk combinado com o próximo (se existir)
            if i < len(chunks) - 1:
                chunk_combinado = chunk + " " + chunks[i + 1]
                if len(chunk_combinado) < 1000:  # Evitar chunks muito longos
                    chunks_overlap.append(chunk_combinado)
        
        return chunks_overlap
    
    def _limpar_texto(self, texto: str) -> str:
        """Limpeza mínima preservando contexto"""
        # Normalizar espaços e quebras de linha
        texto = re.sub(r'\n+', '\n\n', texto)
        texto = re.sub(r'\s+', ' ', texto)
        
        # Remover apenas caracteres realmente problemáticos
        texto = re.sub(r'[^\w\sáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ.,!?;:()\-"]', '', texto)
        
        return texto.strip()
    
    def _vetorizar_chunks(self) -> bool:
        """Cria vetores TF-IDF otimizados"""
        self._log("🔤 Criando vetores TF-IDF...")
        
        try:
            # Configuração otimizada do TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=2000,      # Vocabulário amplo
                ngram_range=(1, 3),     # Uni, bi e trigramas
                stop_words=None,        # Não remover stop words
                lowercase=True,
                min_df=1,               # Frequência mínima
                max_df=0.95,           # Frequência máxima
                token_pattern=r'\b\w+\b',  # Qualquer palavra
                norm='l2',              # Normalização L2
                use_idf=True,           # Usar IDF
                smooth_idf=True,        # Suavizar IDF
                sublinear_tf=True       # Usar log(tf) + 1
            )
            
            # Pré-processar chunks minimamente
            chunks_processados = [self._preprocessar_chunk(chunk) for chunk in self.text_chunks]
            
            # Criar matriz de vetores
            self.chunk_vectors = self.vectorizer.fit_transform(chunks_processados)
            
            # Estatísticas
            vocab_size = len(self.vectorizer.vocabulary_)
            matrix_shape = self.chunk_vectors.shape
            density = self.chunk_vectors.nnz / (matrix_shape[0] * matrix_shape[1])
            
            self._log(f"✅ Vocabulário: {vocab_size} termos")
            self._log(f"📊 Matriz: {matrix_shape} (densidade: {density:.3f})")
            
            return True
            
        except Exception as e:
            self._log(f"Erro na vetorização: {e}")
            return False
    
    def _preprocessar_chunk(self, chunk: str) -> str:
        """Pré-processamento mínimo para manter contexto"""
        # Apenas conversão para minúsculas e normalização de espaços
        chunk = chunk.lower()
        chunk = re.sub(r'\s+', ' ', chunk)
        return chunk.strip()
    
    def consultar(self, pergunta: str, debug_detalhado=False) -> Dict:
        """Consulta principal do sistema"""
        if debug_detalhado:
            self._log(f"🔍 CONSULTA: {pergunta}")
        
        # Pré-processar pergunta
        pergunta_proc = self._preprocessar_chunk(pergunta)
        
        # Vetorizar pergunta
        try:
            vetor_pergunta = self.vectorizer.transform([pergunta_proc])
        except Exception as e:
            self._log(f"Erro na vetorização da pergunta: {e}")
            return self._resposta_erro()
        
        # Calcular similaridades
        similaridades = cosine_similarity(vetor_pergunta, self.chunk_vectors).flatten()
        
        # Estatísticas de similaridade
        max_sim = np.max(similaridades) if len(similaridades) > 0 else 0
        mean_sim = np.mean(similaridades) if len(similaridades) > 0 else 0
        
        if debug_detalhado:
            self._log(f"📊 Sim. máxima: {max_sim:.4f}, média: {mean_sim:.4f}")
        
        # Busca com limiar adaptativo
        chunks_relevantes = self._buscar_chunks_relevantes(similaridades, debug_detalhado)
        
        # Gerar resposta
        resposta_data = self._gerar_resposta(pergunta, chunks_relevantes, max_sim)
        
        # Registrar no histórico
        self.conversation_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'pergunta': pergunta,
            'resposta': resposta_data['resposta'][:100] + "..." if len(resposta_data['resposta']) > 100 else resposta_data['resposta'],
            'chunks_encontrados': len(chunks_relevantes),
            'similaridade_max': max_sim,
            'confianca': resposta_data['confianca']
        })
        
        return resposta_data
    
    def _buscar_chunks_relevantes(self, similaridades: np.ndarray, debug: bool = False) -> List[Dict]:
        """Busca chunks relevantes com estratégias múltiplas"""
        chunks_relevantes = []
        
        # Estratégia 1: Limiar fixo
        for i, sim in enumerate(similaridades):
            if sim >= self.similarity_threshold:
                chunks_relevantes.append({
                    'indice': i,
                    'texto': self.text_chunks[i],
                    'similaridade': sim
                })
        
        # Estratégia 2: Limiar adaptativo se não encontrou nada
        if not chunks_relevantes and np.max(similaridades) > 0:
            limiar_adaptativo = max(self.min_adaptive_threshold, np.max(similaridades) * 0.3)
            
            if debug:
                self._log(f"🔧 Usando limiar adaptativo: {limiar_adaptativo:.4f}")
            
            for i, sim in enumerate(similaridades):
                if sim >= limiar_adaptativo:
                    chunks_relevantes.append({
                        'indice': i,
                        'texto': self.text_chunks[i],
                        'similaridade': sim
                    })
        
        # Estratégia 3: Top-K absoluto como último recurso
        if not chunks_relevantes:
            if debug:
                self._log("🔧 Usando top-3 absoluto")
            
            indices_ordenados = np.argsort(similaridades)[::-1]
            for i in indices_ordenados[:3]:
                if similaridades[i] > 0:
                    chunks_relevantes.append({
                        'indice': i,
                        'texto': self.text_chunks[i],
                        'similaridade': similaridades[i]
                    })
        
        # Ordenar por similaridade
        chunks_relevantes.sort(key=lambda x: x['similaridade'], reverse=True)
        
        if debug:
            self._log(f"✅ Encontrados {len(chunks_relevantes)} chunks relevantes")
        
        return chunks_relevantes[:self.top_chunks]
    
    def _gerar_resposta(self, pergunta: str, chunks: List[Dict], sim_max: float) -> Dict:
        """Gera resposta baseada nos chunks encontrados"""
        if not chunks:
            return {
                'resposta': "Não encontrei informações específicas sobre sua pergunta no texto de 'O Guarani'. Tente reformular a pergunta ou ser mais específico sobre personagens, eventos ou temas da obra.",
                'confianca': 'N/A',
                'chunks_usados': 0,
                'similaridade_max': 0
            }
        
        # Combinar informações dos melhores chunks
        if len(chunks) == 1:
            texto_resposta = chunks[0]['texto']
            intro = "Com base no texto de 'O Guarani':\n\n"
        else:
            # Combinar múltiplos chunks relevantes
            textos = [chunk['texto'] for chunk in chunks[:2]]
            texto_resposta = " ".join(textos)
            intro = "Combinando informações de 'O Guarani':\n\n"
        
        # Truncar se muito longo
        if len(texto_resposta) > 600:
            texto_resposta = texto_resposta[:600] + "..."
        
        resposta_final = intro + texto_resposta
        
        # Determinar confiança
        if sim_max > 0.4:
            confianca = "Alta"
        elif sim_max > 0.15:
            confianca = "Moderada"
        else:
            confianca = "Baixa"
        
        return {
            'resposta': resposta_final,
            'confianca': confianca,
            'chunks_usados': len(chunks),
            'similaridade_max': sim_max
        }
    
    def _resposta_erro(self) -> Dict:
        """Resposta padrão para erros"""
        return {
            'resposta': "Ocorreu um erro ao processar sua pergunta. Tente novamente.",
            'confianca': 'Erro',
            'chunks_usados': 0,
            'similaridade_max': 0
        }
    
    def chat_interativo(self):
        """Interface de chat interativo melhorada"""
        print("\n" + "="*70)
        print("💬 CHATBOT O GUARANI - VERSÃO OTIMIZADA")
        print("Especialista na obra de José de Alencar")
        print("="*70)
        print("📚 Perguntas sugeridas:")
        print("  • Como evolui a relação entre Peri e Dom Antônio?")
        print("  • Quem é Cecília e qual sua importância?")
        print("  • Descreva os personagens principais")
        print("  • Qual é o enredo da história?")
        print("\n💡 Digite 'sair' para encerrar, 'historico' para ver conversas anteriores")
        print("="*70)
        
        while True:
            try:
                pergunta = input("\n🙋 Sua pergunta: ").strip()
                
                if not pergunta:
                    continue
                
                if pergunta.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Obrigado por usar o Chatbot O Guarani!")
                    break
                
                if pergunta.lower() == 'historico':
                    self._mostrar_historico()
                    continue
                
                if pergunta.lower() == 'debug':
                    debug = input("Ativar debug detalhado? (s/n): ").lower().startswith('s')
                    self.debug = debug
                    print(f"Debug {'ativado' if debug else 'desativado'}")
                    continue
                
                # Processar pergunta
                resultado = self.consultar(pergunta, debug_detalhado=True)
                
                # Exibir resposta
                print(f"\n🤖 Resposta:")
                print(resultado['resposta'])
                print(f"\n📊 Confiança: {resultado['confianca']} | ")
                print(f"Chunks: {resultado['chunks_usados']} | ")
                print(f"Similaridade: {resultado['similaridade_max']:.3f}")
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando chat...")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
    
    def _mostrar_historico(self):
        """Mostra histórico de conversas"""
        if not self.conversation_history:
            print("📝 Nenhuma conversa no histórico ainda.")
            return
        
        print("\n" + "="*50)
        print("📚 HISTÓRICO DE CONVERSAS")
        print("="*50)
        
        for i, conv in enumerate(self.conversation_history[-10:], 1):  # Últimas 10
            print(f"\n{i}. [{conv['timestamp']}] {conv['pergunta']}")
            print(f"   🤖 {conv['resposta']}")
            print(f"   📊 {conv['confianca']} | Chunks: {conv['chunks_encontrados']} | Sim: {conv['similaridade_max']:.3f}")
    
    def teste_sistema(self):
        """Executa bateria de testes do sistema"""
        perguntas_teste = [
            "Como evolui a relação entre Peri e Dom Antônio ao longo da história?",
            "Quem é Peri e quais suas características?",
            "Fale sobre Cecília e sua importância na obra",
            "Qual é o enredo principal de O Guarani?",
            "Quem são os aimorés e qual seu papel?",
            "Descreva a natureza brasileira na obra",
            "Como José de Alencar retrata o amor entre Peri e Cecília?",
            "Qual o simbolismo da obra?"
        ]
        
        print("\n🧪 EXECUTANDO BATERIA DE TESTES")
        print("="*60)
        
        sucessos = 0
        for i, pergunta in enumerate(perguntas_teste, 1):
            print(f"\n📋 TESTE {i}: {pergunta}")
            resultado = self.consultar(pergunta, debug_detalhado=False)
            
            # Avaliar sucesso
            sucesso = (resultado['chunks_usados'] > 0 and 
                      resultado['confianca'] != 'N/A' and 
                      len(resultado['resposta']) > 50)
            
            if sucesso:
                sucessos += 1
                print(f"✅ SUCESSO")
            else:
                print(f"❌ FALHOU")
            
            print(f"   📊 Confiança: {resultado['confianca']}")
            print(f"   🔍 Chunks: {resultado['chunks_usados']}")
            print(f"   📈 Similaridade: {resultado['similaridade_max']:.3f}")
            print(f"   🤖 Resposta: {resultado['resposta'][:80]}...")
        
        # Estatísticas finais
        taxa_sucesso = (sucessos / len(perguntas_teste)) * 100
        print(f"\n📈 RESULTADOS FINAIS:")
        print(f"   Testes executados: {len(perguntas_teste)}")
        print(f"   Sucessos: {sucessos}")
        print(f"   Taxa de sucesso: {taxa_sucesso:.1f}%")
        
        return taxa_sucesso
    
    def diagnosticar_sistema(self):
        """Diagnóstica o sistema e mostra estatísticas"""
        print("\n🔧 DIAGNÓSTICO DO SISTEMA")
        print("="*50)
        
        print(f"📖 Texto original: {len(self.original_text):,} caracteres")
        print(f"📝 Total de chunks: {len(self.text_chunks)}")
        
        if self.chunk_vectors is not None:
            print(f"🔤 Vocabulário: {len(self.vectorizer.vocabulary_):,} termos")
            print(f"📊 Matriz de vetores: {self.chunk_vectors.shape}")
            densidade = self.chunk_vectors.nnz / (self.chunk_vectors.shape[0] * self.chunk_vectors.shape[1])
            print(f"📈 Densidade da matriz: {densidade:.4f}")
        
        print(f"⚙️  Limiar de similaridade: {self.similarity_threshold}")
        print(f"🎯 Top chunks retornados: {self.top_chunks}")
        print(f"💬 Conversas no histórico: {len(self.conversation_history)}")
        print(f"📝 Logs de processamento: {len(self.processing_log)}")
        
        # Testar consulta simples
        print(f"\n🧪 Teste rápido:")
        resultado = self.consultar("Quem é Peri?", debug_detalhado=False)
        print(f"   Status: {'✅ OK' if resultado['chunks_usados'] > 0 else '❌ Falhou'}")
        print(f"   Confiança: {resultado['confianca']}")


def main():
    """Função principal para execução"""
    print("🚀 CHATBOT O GUARANI - VERSÃO OTIMIZADA")
    print("="*60)
    
    # Inicializar chatbot
    chatbot = GuaraniChatbotOtimizado(debug=True)
    
    # Menu de opções
    while True:
        print(f"\n🎯 OPÇÕES DISPONÍVEIS:")
        print("1. Chat interativo")
        print("2. Teste do sistema")
        print("3. Diagnóstico completo")
        print("4. Consulta única")
        print("5. Ver histórico")
        print("6. Sair")
        
        try:
            opcao = input("\nEscolha uma opção (1-6): ").strip()
            
            if opcao == '1':
                chatbot.chat_interativo()
            
            elif opcao == '2':
                taxa_sucesso = chatbot.teste_sistema()
                if taxa_sucesso >= 80:
                    print("🎉 Sistema funcionando bem!")
                elif taxa_sucesso >= 60:
                    print("⚠️  Sistema funcional mas pode melhorar")
                else:
                    print("❌ Sistema precisa de ajustes")
            
            elif opcao == '3':
                chatbot.diagnosticar_sistema()
            
            elif opcao == '4':
                pergunta = input("Digite sua pergunta: ").strip()
                if pergunta:
                    resultado = chatbot.consultar(pergunta, debug_detalhado=True)
                    print(f"\n🤖 Resposta:")
                    print(resultado['resposta'])
                    print(f"\n📊 Confiança: {resultado['confianca']}")
            
            elif opcao == '5':
                chatbot._mostrar_historico()
            
            elif opcao == '6':
                print("👋 Até logo!")
                break
            
            else:
                print("❌ Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n👋 Encerrando...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")


# Execução rápida para teste
def teste_rapido():
    """Execução rápida para validação"""
    print("⚡ TESTE RÁPIDO DO SISTEMA")
    print("="*40)
    
    chatbot = GuaraniChatbotOtimizado(debug=False)
    
    # Testar pergunta problemática original
    pergunta_teste = "Como evolui a relação entre Peri e Dom Antonio ao longa da historia?"
    print(f"\n🔍 Testando: {pergunta_teste}")
    
    resultado = chatbot.consultar(pergunta_teste, debug_detalhado=True)
    
    print(f"\n✅ Resultado:")
    print(f"   Chunks encontrados: {resultado['chunks_usados']}")
    print(f"   Confiança: {resultado['confianca']}")
    print(f"   Similaridade: {resultado['similaridade_max']:.3f}")
    print(f"\n🤖 Resposta:")
    print(resultado['resposta'])
    
    return chatbot


if __name__ == "__main__":
    import sys
    
    # Verificar argumentos
    if len(sys.argv) > 1 and sys.argv[1] == "--teste":
        teste_rapido()
    else:
        main()