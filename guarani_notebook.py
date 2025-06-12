# %% [markdown]
# # Chatbot "O Guarani" - Execução e Teste Completo
# 
# Este notebook implementa e testa o sistema de chatbot especializado em "O Guarani" de José de Alencar,
# seguindo as 5 fases descritas no projeto original.

# %% [markdown]
# ## 📋 Preparação Inicial

# %%
# Imports e configurações iniciais
import sys
import os
from datetime import datetime

# Configuração do histórico
historico_execucao = []

def log_execucao(fase, acao, resultado=None):
    """Registra cada passo da execução"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entrada = {
        'timestamp': timestamp,
        'fase': fase,
        'acao': acao,
        'resultado': resultado,
        'status': 'OK' if resultado is not None else 'EXECUTANDO'
    }
    historico_execucao.append(entrada)
    print(f"[{timestamp}] {fase} - {acao} {('✅' if resultado else '⏳')}")

log_execucao("PREP", "Iniciando sistema de histórico")

# %% [markdown]
# ## 🚀 Fase 1: Preparação do Ambiente

# %%
log_execucao("FASE1", "Verificando dependências")

# Instalação e importação das bibliotecas
try:
    import numpy as np
    import pandas as pd
    import re
    from typing import List, Dict, Tuple
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    log_execucao("FASE1", "Bibliotecas básicas importadas", True)
except ImportError as e:
    log_execucao("FASE1", f"Erro na importação: {e}", False)

# %%
# Configuração do NLTK
log_execucao("FASE1", "Configurando NLTK")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import RSLPStemmer
    
    # Downloads necessários
    recursos_nltk = ['punkt', 'stopwords', 'rslp']
    for recurso in recursos_nltk:
        try:
            nltk.data.find(f'tokenizers/{recurso}' if recurso == 'punkt' else f'corpora/{recurso}')
        except LookupError:
            nltk.download(recurso, quiet=True)
    
    log_execucao("FASE1", "NLTK configurado com sucesso", True)
except Exception as e:
    log_execucao("FASE1", f"Erro no NLTK: {e}", False)

# %% [markdown]
# ## 📚 Fase 2: Preparação e Processamento dos Dados

# %%
log_execucao("FASE2", "Carregando texto de O Guarani")

# Texto exemplo expandido da obra (em um projeto real, seria carregado de arquivo)
texto_guarani = """
O Guarani é um romance de José de Alencar publicado em 1857. A história se passa no século XVII, 
durante a colonização do Brasil. O protagonista é Peri, um índio goitacá de força excepcional 
e lealdade inquebrantável.

Peri é caracterizado por sua devoção a Cecília, filha do fidalgo português Dom Antônio de Mariz. 
A relação entre Peri e Ceci representa o encontro entre duas culturas: a indígena e a europeia.
Peri demonstra uma força física impressionante e habilidades de caça excepcionais. Sua lealdade 
a Ceci é absoluta, chegando ao ponto de sacrificar sua própria vida por ela.

Dom Antônio de Mariz é um nobre português que se estabeleceu no Brasil com sua família. 
Ele possui um castelo fortificado às margens do rio Paquequer, onde vive com sua esposa, 
filhos e alguns agregados. Dom Antônio é um homem honrado que respeita Peri pela sua nobreza de caráter.

Cecília, conhecida como Ceci, é a filha de Dom Antônio de Mariz. Ela é descrita como uma jovem 
bela e bondosa, que desenvolve um sentimento especial por Peri. Ceci representa a pureza e 
a inocência, contrastando com a rudeza do ambiente colonial.

A obra retrata os conflitos entre diferentes grupos: os portugueses colonizadores, 
os índios aimorés (inimigos de Peri) e os aventureiros que buscam ouro na região.
Os aimorés são apresentados como uma tribo guerreira e feroz, representando uma ameaça 
constante aos habitantes do castelo.

Álvaro é um jovem português, primo de Cecília, que também habita o castelo. 
Ele representa o europeu civilizado, contrastando com a natureza selvagem de Peri.
Álvaro é corajoso e leal, desenvolvendo uma relação de respeito mútuo com o índio.

Isabel é irmã de Cecília, uma jovem impetuosa que se apaixona por Álvaro. 
Sua história pessoal adiciona complexidade aos relacionamentos na narrativa.
Isabel é caracterizada por sua personalidade forte e determinada.

A natureza brasileira é quase um personagem na obra, sendo descrita com detalhes 
que evidenciam a visão romântica de Alencar sobre a paisagem nacional.
As descrições incluem a exuberante floresta tropical, os rios caudalosos e 
a fauna diversificada da região.

O romance explora temas como o amor impossível entre Peri e Ceci, 
a lealdade, o sacrifício e o choque entre civilizações. Alencar retrata 
o índio como um "bom selvagem", idealizando sua pureza moral.

A linguagem de Alencar mescla o português culto com expressões que buscam 
retratar a fala dos personagens indígenas, criando um estilo único.
O autor utiliza um registro elevado, característico do Romantismo brasileiro.

O enredo culmina com a destruição do castelo e a fuga de Peri e Ceci,
simbolizando o nascimento de uma nova raça brasileira através da união
entre o elemento indígena e o europeu.
"""

print(f"Texto carregado: {len(texto_guarani)} caracteres")
print(f"Aproximadamente {len(texto_guarani.split())} palavras")
log_execucao("FASE2", "Texto carregado com sucesso", len(texto_guarani))

# %%
log_execucao("FASE2", "Iniciando pré-processamento")

# Classe para processamento de texto
class ProcessadorTexto:
    def __init__(self):
        self.stemmer = RSLPStemmer()
        self.stop_words = set(stopwords.words('portuguese'))
        
    def limpar_texto(self, texto):
        """Limpa e normaliza o texto"""
        # Remove quebras de linha excessivas
        texto = re.sub(r'\n+', ' ', texto)
        # Remove espaços múltiplos
        texto = re.sub(r'\s+', ' ', texto)
        # Mantém apenas caracteres alfanuméricos e pontuação básica
        texto = re.sub(r'[^\w\s.,!?;:]', '', texto)
        return texto.strip()
    
    def preprocessar(self, texto):
        """Aplica pré-processamento completo"""
        # Tokenização
        tokens = word_tokenize(texto.lower(), language='portuguese')
        
        # Filtragem: remove stopwords e tokens curtos
        tokens_filtrados = [
            token for token in tokens 
            if (token not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha())
        ]
        
        # Stemming
        tokens_stemmed = [self.stemmer.stem(token) for token in tokens_filtrados]
        
        return ' '.join(tokens_stemmed)
    
    def criar_chunks(self, texto, tamanho_chunk=200, sobreposicao=0.5):
        """Cria chunks de texto com sobreposição"""
        sentencas = sent_tokenize(texto, language='portuguese')
        chunks = []
        
        chunk_atual = []
        contador_palavras = 0
        
        for sentenca in sentencas:
            palavras = sentenca.split()
            
            if contador_palavras + len(palavras) <= tamanho_chunk:
                chunk_atual.append(sentenca)
                contador_palavras += len(palavras)
            else:
                if chunk_atual:
                    chunks.append(' '.join(chunk_atual))
                
                # Aplicar sobreposição
                num_overlap = int(len(chunk_atual) * sobreposicao)
                chunk_atual = chunk_atual[-num_overlap:] if num_overlap > 0 else []
                chunk_atual.append(sentenca)
                contador_palavras = sum(len(s.split()) for s in chunk_atual)
        
        # Adicionar último chunk
        if chunk_atual:
            chunks.append(' '.join(chunk_atual))
        
        return chunks

# Instanciar processador
processador = ProcessadorTexto()

# Limpar texto
texto_limpo = processador.limpar_texto(texto_guarani)
log_execucao("FASE2", "Texto limpo", len(texto_limpo))

# Criar chunks
chunks = processador.criar_chunks(texto_limpo)
print(f"Criados {len(chunks)} chunks de texto")
log_execucao("FASE2", f"Chunks criados: {len(chunks)}", len(chunks))

# %%
# Visualizar exemplos de chunks
log_execucao("FASE2", "Exibindo exemplos de chunks")

print("📋 EXEMPLOS DE CHUNKS CRIADOS:")
print("=" * 60)

for i, chunk in enumerate(chunks[:3]):  # Mostra apenas os 3 primeiros
    print(f"\n🔹 Chunk {i+1}:")
    print(f"   Tamanho: {len(chunk.split())} palavras")
    print(f"   Texto: {chunk[:150]}...")

# %%
# Pré-processar chunks
log_execucao("FASE2", "Pré-processando chunks")

chunks_processados = []
for i, chunk in enumerate(chunks):
    chunk_processado = processador.preprocessar(chunk)
    chunks_processados.append(chunk_processado)

print(f"Chunks pré-processados: {len(chunks_processados)}")
log_execucao("FASE2", "Pré-processamento concluído", len(chunks_processados))

# Exemplo de chunk antes e depois do processamento
print("\n📊 EXEMPLO DE PROCESSAMENTO:")
print("Antes:", chunks[0][:200] + "...")
print("Depois:", chunks_processados[0][:200] + "...")

# %% [markdown]
# ## 🗄️ Fase 3: Armazenamento e Indexação

# %%
log_execucao("FASE3", "Criando vectorizador TF-IDF")

# Configurar vectorizador TF-IDF
vectorizador = TfidfVectorizer(
    max_features=1000,  # Máximo de features
    ngram_range=(1, 2),  # Uni e bigramas
    min_df=1,  # Frequência mínima de documento
    max_df=0.95  # Frequência máxima de documento
)

# Treinar vectorizador e criar matriz de vetores
matriz_vetores = vectorizador.fit_transform(chunks_processados)

print(f"Matriz de vetores: {matriz_vetores.shape}")
print(f"Vocabulário: {len(vectorizador.vocabulary_)} termos")

log_execucao("FASE3", "Vetorização concluída", matriz_vetores.shape[0])

# %%
# Analisar vocabulário mais importante
log_execucao("FASE3", "Analisando vocabulário")

# Obter nomes das features
feature_names = vectorizador.get_feature_names_out()

# Calcular pontuação TF-IDF média para cada termo
pontuacoes_medias = np.array(matriz_vetores.mean(axis=0)).flatten()

# Criar DataFrame para análise
vocab_df = pd.DataFrame({
    'termo': feature_names,
    'tfidf_medio': pontuacoes_medias
}).sort_values('tfidf_medio', ascending=False)

print("🔝 TOP 10 TERMOS MAIS RELEVANTES:")
print(vocab_df.head(10))

log_execucao("FASE3", "Análise de vocabulário concluída", len(feature_names))

# %% [markdown]
# ## 🔍 Fase 4: Sistema de Busca e Resposta

# %%
log_execucao("FASE4", "Implementando sistema de busca")

class SistemaBusca:
    def __init__(self, chunks_originais, chunks_processados, vectorizador, matriz_vetores):
        self.chunks_originais = chunks_originais
        self.chunks_processados = chunks_processados
        self.vectorizador = vectorizador
        self.matriz_vetores = matriz_vetores
        self.processador = ProcessadorTexto()
        self.limiar_similaridade = 0.1
        self.top_k = 3
    
    def buscar(self, pergunta):
        """Busca chunks relevantes para a pergunta"""
        # Pré-processar pergunta
        pergunta_processada = self.processador.preprocessar(pergunta)
        
        # Vetorizar pergunta
        vetor_pergunta = self.vectorizador.transform([pergunta_processada])
        
        # Calcular similaridades
        similaridades = cosine_similarity(vetor_pergunta, self.matriz_vetores).flatten()
        
        # Filtrar e ordenar resultados
        resultados = []
        for i, sim in enumerate(similaridades):
            if sim >= self.limiar_similaridade:
                resultados.append({
                    'chunk_id': i,
                    'texto_original': self.chunks_originais[i],
                    'similaridade': sim
                })
        
        # Ordenar por similaridade
        resultados.sort(key=lambda x: x['similaridade'], reverse=True)
        
        return resultados[:self.top_k]
    
    def gerar_resposta(self, pergunta, resultados):
        """Gera resposta baseada nos resultados da busca"""
        if not resultados:
            return "Desculpe, não encontrei informações relevantes sobre sua pergunta no texto de 'O Guarani'."
        
        # Usar o chunk mais relevante como base da resposta
        melhor_resultado = resultados[0]
        texto_base = melhor_resultado['texto_original']
        confianca = melhor_resultado['similaridade']
        
        # Formatação da resposta
        resposta = f"Com base em 'O Guarani':\n\n{texto_base}"
        
        # Adicionar indicador de confiança
        if confianca > 0.5:
            resposta += "\n\n(Resposta com alta confiança)"
        elif confianca > 0.3:
            resposta += "\n\n(Resposta com confiança moderada)"
        else:
            resposta += "\n\n(Resposta com baixa confiança)"
        
        return resposta

# Instanciar sistema de busca
sistema_busca = SistemaBusca(chunks, chunks_processados, vectorizador, matriz_vetores)

log_execucao("FASE4", "Sistema de busca implementado", True)

# %%
# Testar sistema com perguntas exemplo
log_execucao("FASE4", "Testando sistema com perguntas exemplo")

perguntas_teste = [
    "Quem é Peri?",
    "Fale sobre Cecília",
    "Qual é o enredo de O Guarani?",
    "Quem são os personagens principais?",
    "Onde se passa a história?",
    "Como é descrita a natureza no livro?"
]

resultados_teste = []

print("🧪 TESTE DO SISTEMA DE BUSCA")
print("=" * 50)

for pergunta in perguntas_teste:
    print(f"\n❓ Pergunta: {pergunta}")
    
    # Buscar
    resultados = sistema_busca.buscar(pergunta)
    
    # Gerar resposta
    resposta = sistema_busca.gerar_resposta(pergunta, resultados)
    
    # Exibir resultado resumido
    if resultados:
        sim_max = max([r['similaridade'] for r in resultados])
        print(f"🎯 Similaridade máxima: {sim_max:.3f}")
        print(f"📝 Chunks encontrados: {len(resultados)}")
        print(f"🤖 Resposta: {resposta[:100]}...")
    else:
        print("❌ Nenhum resultado encontrado")
    
    # Armazenar para histórico
    resultados_teste.append({
        'pergunta': pergunta,
        'resposta': resposta,
        'num_resultados': len(resultados),
        'similaridade_max': max([r['similaridade'] for r in resultados]) if resultados else 0
    })

log_execucao("FASE4", f"Teste concluído com {len(perguntas_teste)} perguntas", len(perguntas_teste))

# %% [markdown]
# ## 🎯 Fase 5: Interface e Demonstração

# %%
log_execucao("FASE5", "Preparando interface de demonstração")

def demonstrar_consulta_detalhada(pergunta):
    """Demonstra uma consulta com detalhes completos"""
    print(f"🔍 ANÁLISE DETALHADA DA CONSULTA")
    print("=" * 60)
    print(f"Pergunta: {pergunta}")
    print()
    
    # Buscar resultados
    resultados = sistema_busca.buscar(pergunta)
    
    # Mostrar processo de busca
    pergunta_processada = sistema_busca.processador.preprocessar(pergunta)
    print(f"Pergunta processada: {pergunta_processada}")
    print()
    
    # Mostrar resultados encontrados
    print(f"Resultados encontrados: {len(resultados)}")
    print()
    
    for i, resultado in enumerate(resultados):
        print(f"📄 Resultado {i+1}:")
        print(f"   Similaridade: {resultado['similaridade']:.3f}")
        print(f"   Texto: {resultado['texto_original'][:200]}...")
        print()
    
    # Gerar e mostrar resposta final
    resposta = sistema_busca.gerar_resposta(pergunta, resultados)
    print("🤖 RESPOSTA FINAL:")
    print("=" * 40)
    print(resposta)
    
    return resposta

# Demonstração com pergunta específica
pergunta_demo = "Quem é Peri e quais são suas principais características?"
resposta_demo = demonstrar_consulta_detalhada(pergunta_demo)

log_execucao("FASE5", "Demonstração detalhada concluída", True)

# %% [markdown]
# ## 📊 Análise de Performance e Histórico

# %%
log_execucao("ANÁLISE", "Gerando estatísticas do sistema")

# Estatísticas gerais
print("📈 ESTATÍSTICAS DO SISTEMA")
print("=" * 50)
print(f"Total de chunks: {len(chunks)}")
print(f"Vocabulário: {len(vectorizador.vocabulary_)} termos")
print(f"Dimensões da matriz: {matriz_vetores.shape}")
print(f"Densidade da matriz: {matriz_vetores.nnz / (matriz_vetores.shape[0] * matriz_vetores.shape[1]):.3f}")

# Análise dos testes
if resultados_teste:
    similaridades = [r['similaridade_max'] for r in resultados_teste]
    print(f"\nPerguntas testadas: {len(resultados_teste)}")
    print(f"Similaridade média: {np.mean(similaridades):.3f}")
    print(f"Similaridade máxima: {np.max(similaridades):.3f}")
    print(f"Similaridade mínima: {np.min(similaridades):.3f}")

log_execucao("ANÁLISE", "Estatísticas geradas", len(resultados_teste))

# %%
# Visualização das similaridades
log_execucao("ANÁLISE", "Criando visualizações")

if resultados_teste:
    plt.figure(figsize=(12, 6))
    
    # Gráfico de barras das similaridades
    plt.subplot(1, 2, 1)
    perguntas_abrev = [r['pergunta'][:20] + "..." for r in resultados_teste]
    similaridades = [r['similaridade_max'] for r in resultados_teste]
    
    plt.bar(range(len(similaridades)), similaridades)
    plt.title('Similaridade por Pergunta')
    plt.xlabel('Pergunta')
    plt.ylabel('Similaridade Máxima')
    plt.xticks(range(len(perguntas_abrev)), perguntas_abrev, rotation=45, ha='right')
    
    # Histograma das similaridades
    plt.subplot(1, 2, 2)
    plt.hist(similaridades, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribuição das Similaridades')
    plt.xlabel('Similaridade')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.show()
    
    log_execucao("ANÁLISE", "Visualizações criadas", True)

# %%
# Histórico completo de execução
log_execucao("HISTÓRICO", "Compilando histórico completo")

print("📋 HISTÓRICO COMPLETO DE EXECUÇÃO")
print("=" * 70)

# Agrupar por fase
fases = {}
for entrada in historico_execucao:
    fase = entrada['fase']
    if fase not in fases:
        fases[fase] = []
    fases[fase].append(entrada)

# Exibir por fase
for fase, entradas in fases.items():
    print(f"\n🔹 {fase}:")
    for entrada in entradas:
        status = "✅" if entrada['status'] == 'OK' else "⏳" if entrada['status'] == 'EXECUTANDO' else "❌"
        resultado_texto = f" → {entrada['resultado']}" if entrada['resultado'] is not None else ""
        print(f"   [{entrada['timestamp']}] {status} {entrada['acao']}{resultado_texto}")

# Resumo final
total_acoes = len(historico_execucao)
acoes_ok = len([e for e in historico_execucao if e['status'] == 'OK'])
percentual_sucesso = (acoes_ok / total_acoes) * 100 if total_acoes > 0 else 0

print(f"\n📊 RESUMO FINAL:")
print(f"   Total de ações: {total_acoes}")
print(f"   Ações bem-sucedidas: {acoes_ok}")
print(f"   Taxa de sucesso: {percentual_sucesso:.1f}%")

log_execucao("HISTÓRICO", "Histórico compilado", total_acoes)

# %% [markdown]
# ## 🎉 Sistema Completo Implementado
# 
# O chatbot "O Guarani" foi implementado com sucesso seguindo todas as 5 fases:
# 
# 1. ✅ **Preparação do Ambiente**: Bibliotecas configuradas
# 2. ✅ **Processamento dos Dados**: Texto limpo e segmentado em chunks
# 3. ✅ **Armazenamento e Indexação**: Vetores TF-IDF criados
# 4. ✅ **Sistema de Busca**: Busca por similaridade implementada
# 5. ✅ **Interface**: Sistema testado e demonstrado
# 
# ### Próximos Passos:
# - Expandir base de texto com obra completa
# - Implementar Word2Vec para melhor semântica
# - Adicionar interface web com Streamlit
# - Melhorar geração de respostas com templates
# - Adicionar avaliação automática de qualidade

print("\n🎉 SISTEMA CHATBOT 'O GUARANI' IMPLEMENTADO COM SUCESSO!")
print("Todas as fases foram executadas e testadas.")
print("Histórico completo de execução mantido e disponível.")
