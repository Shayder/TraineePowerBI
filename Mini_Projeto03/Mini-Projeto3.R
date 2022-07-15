# Mini-Projeto 3
# Prevendo a Inadiplência de Clientes com Machine Learning e Power BI


# Definindo pasta de trabalho
setwd("C:/Users/shayd/Desktop/TreinamentoCompletoPowerBi/Cap15")

# Instalando os pacotes para o projeto
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

# Carregando os pacotes
library("Amelia")
library("caret")
library("ggplot2")
library("dplyr")
library("reshape")
library("randomForest")
library("e1071")

# Carregando o dataset
dados_clientes <- read.csv("dados/dataset.csv")

# Visualizando os dados e sua estrutura
View(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

### Análise Exploratória, Limpeza e Transformação ###

# Removendo a primeira coluna ID
dados_clientes$ID <- NULL


# Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "INADIMPLENTE"


# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

# Convertendo os atributos genero, escolaridade, estado civil e idade para fatores(categorias)

# Renomeando colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)

# Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
dados_clientes$Genero <- cut(dados_clientes$Genero, 
                             c(0,1,2), 
                             labels = c("Masculino",
                                        "Feminino"))

# Escolaridade
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade, 
                                   c(0,1,2,3,4), 
                                   labels = c("Pós Graduado",
                                              "Graduado",
                                              "Ensino Médio",
                                              "Outros"))

# Estado Civil
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil, 
                                   c(-1,0,1,2,3), 
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))
summary(dados_clientes$Estado_Civil)

# Convertendo a variável para o tipo com faixa etária
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c("Jovem",
                                       "Adulto",
                                       "Idoso"))

# Convertendo as variáveis "PAY" para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Visualizando dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores MissingObservados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Obserbados")
dim(dados_clientes)
View(dados_clientes)

# Alterando tipo da variável dependente para fator
str(dados_clientes$INADIMPLENTE)
colnames(dados_clientes)
dados_clientes$INADIMPLENTE <- as.factor(dados_clientes$INADIMPLENTE)
str(dados_clientes$INADIMPLENTE)
View(dados_clientes)

# Total de inadimplentes versus não-inadimplentes
table(dados_clientes$INADIMPLENTE)

# Porcentagem entre as classes
prop.table(table(dados_clientes$INADIMPLENTE))

# Plot da distribuição usando ggplot2
qplot(INADIMPLENTE, data = dados_clientes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust =1))

# Set seed
set.seed(12345)

# Amostragem estratificada

# Selecione as linhas de acordo com a variável inadimplente como strata
indice <- createDataPartition(dados_clientes$INADIMPLENTE, p =0.75, list = FALSE)
dim(indice)

# Definimos os dados de treinamento como sibconjunto do conjunto de dados original
# com números de indice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
table(dados_treino$INADIMPLENTE)

# Porcentagem entre as classes
prop.table(table(dados_treino$INADIMPLENTE))

# Número de registros no dataset de treinamento
dim(dados_treino)


# Comparamos as porcentagens entre as classes de treino e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$INADIMPLENTE)),
                       prop.table(table(dados_clientes$INADIMPLENTE)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Tudo o que não está no dataset de treinamento está no dataset de teste. Observe o sinal de menos
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim

### Modelo de Machine Learning ###

# Construindo a primeira versão do modelo
modelo_v1 <- randomForest(INADIMPLENTE ~ ., data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$INADIMPLENTE, positive = "1")
cm_v1

# Calculando Precision, Recalll e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$INADIMPLENTE
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2* precision * recall) / (precision + recall)
F1

# Balanceamento de classe
install.packages('smotefamily')
install.packages("performanceEstimation")
library(performanceEstimation)

# Aplicando o SMOTE -SMOTE: Syntheric Minority Over-sampling Technique

table(dados_treino$INADIMPLENTE)
prop.table(table(dados_treino$INADIMPLENTE))
set.seed(9560)
dados_treino_bal <- smote(INADIMPLENTE ~ ., data = dados_treino)
table(dados_treino_bal$INADIMPLENTE)
prop.table(table(dados_treino_bal$INADIMPLENTE))

# Segunda versão do modelo
modelo_v2 <- randomForest(INADIMPLENTE ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$INADIMPLENTE, positive = "1")
cm_v2

# Calculando Precision, Recalll e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$INADIMPLENTE
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2* precision * recall) / (precision + recall)
F1

# Obtendo as variaveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                     Importance = round(imp_var[ , 'MeanDecreaseGini'], 2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar  importância relativa das vari´veis
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) + 
  geom_bar(stat = 'identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Terceira versõa do modelo apenas com as variáveis mais importante
colnames(dados_treino_bal)
View(dados_treino_bal)
modelo_v3 <- randomForest(INADIMPLENTE ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + 
                            BILL_AMT1, data = dados_treino_bal )
modelo_v3

# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$INADIMPLENTE, positive = "1")
cm_v3

# Calculando Precision, Recalll e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$INADIMPLENTE
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2* precision * recall) / (precision + recall)
F1

# Salvando o modelo em discoveries
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")

# Previsão com dados de 3 novos clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)
