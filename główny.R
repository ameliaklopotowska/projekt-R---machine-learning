library("foreign")
library(plyr)
library(caret)
library(pROC)
library(lares)
library(tidyr)
library(neuralnet)
library(data.tree)
library(rpart)
library(tidyverse)

source("./funkcje.R")
options(warn=-1)
#wczytywanie danych
zbior_wieloklasowa = read.table("./wieloklasowa/balance-scale.data",sep = ",", header = T)
zbior_binarna = read.arff("./binarna/Qualitative_Bankruptcy.arff")
zbior_regresja = read.csv(file = './regresja/forestfires.csv')

#Regresja przygotowanie danych
y_min_regresja <- min( zbior_regresja[['Y']] )
y_max_regresja <- max( zbior_regresja[['Y']] )
# zamienimy zmienne porzadkowe - miesiace i dni tygodnia na liczby
zbior_regresja[["month"]] <- encode_ordinal(zbior_regresja[["month"]], order=c('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))
zbior_regresja[["day"]] <- encode_ordinal(zbior_regresja[["day"]], order=c('mon','tue','wed','thu','fri','sat','sun'))
names(zbior_regresja)[names(zbior_regresja) == "Y"] <- "Point_Y"
names(zbior_regresja)[names(zbior_regresja) == "X"] <- "Point_X"
zbior_regresja <- cbind("Y"=round(as.numeric(zbior_regresja[,13])), zbior_regresja[,c(1:12)])
#ucinam lekko zbior zeby operacje nie wykonywaly sie w nieskonczonosc
zbior_regresja = zbior_regresja[sample(nrow(zbior_regresja), 200), ]
#normalizacja zmiennych
zbior_regresja_norm <- zbior_regresja
zbior_regresja_norm[, 2:11] <- sapply( zbior_regresja[, c(2:11)], MinMax )
colnames(zbior_regresja_norm)
head(zbior_regresja_norm)
count(zbior_regresja_norm, "Y")

#Binarna przygotowanie danych
#zmieniam zmienne porzadkowe i nominalne
zbior_binarna[["IR"]] <- encode_ordinal(zbior_binarna[["IR"]], order=c('N','A','P'))
zbior_binarna[["MR"]] <- encode_ordinal(zbior_binarna[["MR"]], order=c('N','A','P'))
zbior_binarna[["FF"]] <- encode_ordinal(zbior_binarna[["FF"]], order=c('N','A','P'))
zbior_binarna[["CR"]] <- encode_ordinal(zbior_binarna[["CR"]], order=c('N','A','P'))
zbior_binarna[["CO"]] <- encode_ordinal(zbior_binarna[["CO"]], order=c('N','A','P'))
zbior_binarna[["OP"]] <- encode_ordinal(zbior_binarna[["OP"]], order=c('N','A','P'))
zbior_binarna[['Class']] <- ifelse( zbior_binarna[['Class']] == "NB", 0, 1)
zbior_binarna$Class <- as.factor(zbior_binarna$Class)
zbior_binarna <- cbind("Y"=zbior_binarna[,7], zbior_binarna[,1:6])
zbior_binarna <- na.omit(zbior_binarna) 
#zbior przecietnie zbalansowany
count(zbior_binarna, "Y")
head(zbior_binarna)
zbior_binarna_norm <- zbior_binarna
zbior_binarna_norm[, 2:7]  <- sapply( zbior_binarna_norm[, c(2:7)], MinMax )
head(zbior_binarna_norm)

#Wieloklasowa przygotowanie danych
zbior_wieloklasowa$B <- as.factor(zbior_wieloklasowa$B)
zbior_wieloklasowa <- cbind("Y"=zbior_wieloklasowa[,1], zbior_wieloklasowa[,2:5])
zbior_wieloklasowa <- na.omit(zbior_wieloklasowa) 
#ucinam(losuje) zbior zeby operacje nie wykonywaly sie w nieskonczonosc
zbior_wieloklasowa = zbior_wieloklasowa[sample(nrow(zbior_wieloklasowa), 200), ]
count(zbior_wieloklasowa, "Y")
head(zbior_wieloklasowa)
zbior_wieloklasowa_norm <- zbior_wieloklasowa
zbior_wieloklasowa_norm[, 2:5]  <- sapply( zbior_wieloklasowa_norm[, c(2:5)], MinMax )
head(zbior_wieloklasowa_norm)

k=c(2,3,6,9,12)
parTune_knn <- crossing(k = k)
parTune_knn <- data.frame(parTune_knn)

#### KNN ####
#REGRESJA#
res_knn_reg <- CrossValidation_knn_reg(zbior_regresja_norm, 3, 111, parTune_knn)
res_knn_reg <- res_knn_reg[order(res_knn_reg$mape_test),]
print("Trzy najlepsze modele dla KNN regresji: ")
print(head(res_knn_reg,3))

ggplot(res_knn_reg, aes(k)) + 
  geom_point(aes(y = mape_test, colour = "MAPE Test")) + 
  geom_point(aes(y = mape_train, colour = "MAPE Trening")) +
  geom_point(aes(y = mape_caret, colour = "MAPE Caret")) + 
  ylim(0,3) +
  ylab("MAPE") +
  ggtitle("Wplyw 'k' na blad MAPE w modelu KNN dla regresji") +
  labs(colour="Dane")

ggplot(res_knn_reg, aes(k)) + 
  geom_point(aes(y = mse_test, colour = "MSE Test")) + 
  geom_point(aes(y = mse_train, colour = "MSE Trening")) +
  geom_point(aes(y = mse_caret, colour = "MSE Caret")) + 
  ylim(300,4500) +
  ylab("MSE") +
  ggtitle("Wplyw 'k' na blad MSE w modelu KNN dla regresji") +
  labs(colour="Dane")

#BINARNE#
res_knn_bin <- CrossValidation_knn_bin(zbior_binarna_norm, 3, 111, parTune_knn)
res_knn_bin <- res_knn_bin[order(-res_knn_bin$dokladnosc_test),]
print("Trzy najlepsze modele dla KNN klasyfikacji binarnej: ")
print(head(res_knn_bin,3))

ggplot(res_knn_bin, aes(k)) + 
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc Trening")) + 
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc Test")) +
  geom_point(aes(y = dokladnosc_caret, colour = "Dokladnosc Caret")) + 
  ylim(0.98,1) +
  ylab("Dokladnosc") +
  ggtitle("Wplyw 'k' na dokladnosc w modelu KNN dla klasyfikacji binarnej") +
  labs(colour="Dane")

ggplot(res_knn_bin, aes(k)) + 
  geom_point(aes(y = precyzja_train, colour = "Precyzja Trening")) + 
  geom_point(aes(y = precyzja_test, colour = "Precyzja Test")) +
  geom_point(aes(y = precyzja_caret, colour = "Precyzja Caret")) + 
  ylim(0.98,1) +
  ylab("Precyzja") +
  ggtitle("Wplyw 'k' na precyzje w modelu KNN dla klasyfikacji binarnej") +
  labs(colour="Dane")

#WIELOKLASOWA#
res_knn_wiel <- CrossValidation_knn_wiel(zbior_wieloklasowa_norm, 3, 111, parTune_knn)
res_knn_wiel <- res_knn_wiel[order(-res_knn_wiel$dokladnosc_test),]
print("Trzy najlepsze modele dla KNN dla klasyfikacji wieloklasowej: ")
print(head(res_knn_wiel,3))

ggplot(res_knn_wiel, aes(k)) + 
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc Trening")) + 
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc Test")) +
  geom_point(aes(y = dokladnosc_caret, colour = "Dokladnosc Caret")) + 
  ylim(0,1) +
  ylab("Dokladnosc") +
  ggtitle("Wplyw 'k' na dokladnosc w modelu KNN dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

ggplot(res_knn_wiel, aes(k)) + 
  geom_point(aes(y = auc_train, colour = "AUC Trening")) + 
  geom_point(aes(y = auc_test, colour = "AUC Test")) +
  geom_point(aes(y = auc_caret, colour = "AUC Caret")) + 
  ylim(0,1) +
  ylab("AUC") +
  ggtitle("Wplyw 'k' na AUC w modelu KNN dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

par_nn <- crossing(h = list(c(16,8,8,8,4,2), c(32,8,4,2,2), c(16,8,4,2), c(8,4,2), c(32,16)), lr = c(0.01, 0.001, 0.005) , iter = c(200, 2000))
par_nn<- data.frame(par_nn)
#par_nn <- crossing(h = list(  c(8,4,2)), lr = c(0.1) , iter = c(2000))
#par_nn<- data.frame(par_nn)

### SIECI ###
#REGRESJA#
res_nn_reg <- CrossValidation_nn_reg(zbior_regresja_norm, 3, 111, par_nn)
res_nn_reg <- transform(res_nn_reg, mape_train = as.numeric(mape_train), 
               mape_test = as.numeric(mape_test),
               mape_neuralnet = as.numeric(mape_neuralnet),
               mse_train = as.numeric(mse_train), 
               mse_test = as.numeric(mse_test),
               mse_neuralnet = as.numeric(mse_neuralnet))
res_nn_reg <- res_nn_reg[4:35,][order(res_nn_reg[4:28,]$mape_test),]
print("Trzy najlepsze modele dla sieci dla regresji: ")
print(head(res_nn_reg,3))

ggplot(res_nn_reg, aes(hn)) +
  geom_point(aes(y = mape_train, colour = "MAPE train")) +
  geom_point(aes(y = mape_test, colour = "MAPE test")) +
  geom_point(aes(y = mape_neuralnet, colour = "MAPE neuralnet")) + 
  ylim(0, 1) + 
  ylab("MAPE") +
  ggtitle("Wplyw H na blad MAPE w NN dla regresji") +
  labs(colour="Dane")

ggplot(res_nn_reg, aes(hn)) + 
  geom_point(aes(y = mse_test, colour = "MSE Test")) + 
  geom_point(aes(y = mse_train, colour = "MSE Trening")) +
  geom_point(aes(y = mse_neuralnet, colour = "MSE neuralnet")) + 
  ylim(0,1000) +
  ylab("MSE") +
  ggtitle("Wplyw 'h' na blad MSE w modelu NN dla regresji") +
  labs(colour="Dane")

ggplot(res_nn_reg, aes(lr)) +
  geom_point(aes(y = mape_train, colour = "MAPE train")) +
  geom_point(aes(y = mape_test, colour = "MAPE test")) +
  geom_point(aes(y = mape_neuralnet, colour = "MAPE neuralnet")) + 
  ylim(0, 1) + 
  ylab("MAPE") +
  ggtitle("Wplyw lr na MAPE w NN dla regresji") +
  labs(colour="Dane")

#BINARNA#
res_nn_bin <- CrossValidation_nn_bin(zbior_binarna_norm, 3, 111, par_nn)
res_nn_bin <- transform(res_nn_bin, dokladnosc_train = as.numeric(dokladnosc_train), 
                        dokladnosc_test = as.numeric(dokladnosc_test),
                        dokladnosc_neuralnet = as.numeric(dokladnosc_neuralnet),
                        precyzja_train = as.numeric(precyzja_train), 
                        precyzja_test = as.numeric(precyzja_test),
                        precyzja_neuralnet = as.numeric(precyzja_neuralnet))
res_nn_bin <- res_nn_bin[order(-res_nn_bin$dokladnosc_test),]
print("Trzy najlepsze modele dla sieci dla klasyfikacji binarnej: ")
print(head(res_nn_bin,3))

ggplot(res_nn_bin, aes(h)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_neuralnet, colour = "Dokladnosc neuralnet")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw H na dokladnosc w NN dla klasyfikacji binarnej") +
  labs(colour="Dane")

ggplot(res_nn_bin, aes(h)) +
  geom_point(aes(y = precyzja_train, colour = "Precyzja train")) +
  geom_point(aes(y = precyzja_test, colour = "Precyzja test")) +
  geom_point(aes(y = precyzja_neuralnet, colour = "Precyzja neuralnet")) + 
  ylim(0, 1) + 
  ylab("Precyzja") +
  ggtitle("Wplyw H na precyzje w NN dla klasyfikacji binarnej") +
  labs(colour="Dane")

ggplot(res_nn_bin, aes(lr)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_neuralnet, colour = "Dokladnosc neuralnet")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw lr na dokladnosc w NN dla klasyfikacji binarnej") +
  labs(colour="Dane")

#WIELOKLASOWA#
res_nn_wiel <- CrossValidation_nn_wiel(zbior_wieloklasowa_norm, 3, 111, par_nn)
res_nn_wiel <- transform(res_nn_wiel, dokladnosc_train = as.numeric(dokladnosc_train), 
                        dokladnosc_test = as.numeric(dokladnosc_test),
                        dokladnosc_neuralnet = as.numeric(dokladnosc_neuralnet),
                        auc_train = as.numeric(auc_train), 
                        auc_test = as.numeric(auc_test),
                        auc_neuralnet = as.numeric(auc_neuralnet))
res_nn_wiel <- res_nn_wiel[order(-res_nn_wiel$dokladnosc_test),]
print("Trzy najlepsze modele dla sieci dla klasyfikacji wieloklasowej: ")
print(head(res_nn_wiel,3))

ggplot(res_nn_wiel, aes(h)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_neuralnet, colour = "Dokladnosc neuralnet")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw H na dokladnosc w NN dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

ggplot(res_nn_wiel, aes(h)) +
  geom_point(aes(y = auc_train, colour = "AUC train")) +
  geom_point(aes(y = auc_test, colour = "AUC test")) +
  geom_point(aes(y = auc_neuralnet, colour = "AUC neuralnet")) + 
  ylim(0, 1) + 
  ylab("AUC") +
  ggtitle("Wplyw H na AUC w NN dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

ggplot(res_nn_wiel, aes(lr)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_neuralnet, colour = "Dokladnosc neuralnet")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw lr na dokladnosc w NN dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

### DRZEWA ###
#REGRESJA#
par_tree_reg <- crossing(minobs = c(2,5,8), depth = c(4,6,8,10,12), type = c("SS"))
par_tree_reg <- data.frame(par_tree_reg)

res_tree_reg <- CrossValidation_tree_reg(zbior_regresja_norm, 3, 111, par_tree_reg)
res_tree_reg <- transform(res_tree_reg, mape_train = as.numeric(mape_train), 
                        mape_test = as.numeric(mape_test),
                        mape_rpart = as.numeric(mape_rpart),
                        mse_train = as.numeric(mse_train), 
                        mse_test = as.numeric(mse_test),
                        mse_rpart = as.numeric(mse_rpart))
res_tree_reg <- res_tree_reg[order(res_tree_reg$mape_test),]
print("Trzy najlepsze modele dla sieci dla regresji: ")
print(head(res_tree_reg,3))

ggplot(res_tree_reg, aes(minobs)) +
  geom_point(aes(y = mape_train, colour = "MAPE train")) +
  geom_point(aes(y = mape_test, colour = "MAPE test")) +
  geom_point(aes(y = mape_rpart, colour = "MAPE rpart")) + 
  ylim(0, 2) + 
  ylab("MAPE") +
  ggtitle("Wplyw minobs na blad MAPE w drzewach dla regresji") +
  labs(colour="Dane")

ggplot(res_tree_reg, aes(minobs)) + 
  geom_point(aes(y = mse_test, colour = "MSE Test")) + 
  geom_point(aes(y = mse_train, colour = "MSE Trening")) +
  geom_point(aes(y = mse_rpart, colour = "MSE rpart")) + 
  ylim(0,1500) +
  ylab("MSE") +
  ggtitle("Wplyw minobs na blad MSE w drzewach dla regresji") +
  labs(colour="Dane")

ggplot(res_tree_reg, aes(depth)) +
  geom_point(aes(y = mape_train, colour = "MAPE train")) +
  geom_point(aes(y = mape_test, colour = "MAPE test")) +
  geom_point(aes(y = mape_rpart, colour = "MAPE rpart")) + 
  ylim(0, 2) + 
  ylab("MAPE") +
  ggtitle("Wplyw depth na MAPE w drzewach dla regresji") +
  labs(colour="Dane")

#BINARNA#
par_tree_bin <- crossing(minobs = c(2,10,100,50), depth = c(2,4,6,8,10,12,30), type = c("Gini", "Entropy"))
par_tree_bin <- data.frame(par_tree_bin)

res_tree_bin <- CrossValidation_tree_bin(zbior_binarna_norm, 3, 111, par_tree_bin)
res_tree_bin <- transform(res_tree_bin, dokladnosc_train = as.numeric(dokladnosc_train), 
                          dokladnosc_test = as.numeric(dokladnosc_test),
                          dokladnosc_rpart = as.numeric(dokladnosc_rpart),
                          precyzja_train = as.numeric(precyzja_train), 
                          precyzja_test = as.numeric(precyzja_test),
                          precyzja_rpart = as.numeric(precyzja_rpart))
res_tree_bin <- res_tree_bin[order(-res_tree_bin$dokladnosc_test),]
print("Trzy najlepsze modele dla sieci dla regresji: ")
print(head(res_tree_bin,3))

ggplot(res_tree_bin, aes(minobs)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw minobs na dokladnosc w drzewach dla klasyfikacji binarnej") +
  labs(colour="Dane")

ggplot(res_tree_bin, aes(depth)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw depth na dokladnosc w drzewach dla klasyfikacji binarnej") +
  labs(colour="Dane")

ggplot(res_tree_bin, aes(type)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw type na dokladnosc w drzewach dla klasyfikacji binarnej") +
  labs(colour="Dane")

#WIELOKLASOWA#
res_tree_wiel <- CrossValidation_tree_wiel(zbior_wieloklasowa_norm, 3, 111, par_tree_bin)
res_tree_wiel <- transform(res_tree_wiel, dokladnosc_train = as.numeric(dokladnosc_train), 
                          dokladnosc_test = as.numeric(dokladnosc_test),
                          dokladnosc_rpart = as.numeric(dokladnosc_rpart),
                          precyzja_train = as.numeric(auc_train), 
                          precyzja_test = as.numeric(auc_test),
                          precyzja_rpart = as.numeric(auc_rpart))
res_tree_wiel <- res_tree_wiel[order(-res_tree_wiel$dokladnosc_test),]
print("Trzy najlepsze modele dla sieci dla klasyfikacji wieloklasowej: ")
print(head(res_tree_wiel,3))

ggplot(res_tree_wiel, aes(minobs)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw minobs na dokladnosc w drzewach dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

ggplot(res_tree_wiel, aes(depth)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw depth na dokladnosc w drzewach dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")

ggplot(res_tree_wiel, aes(type)) +
  geom_point(aes(y = dokladnosc_train, colour = "Dokladnosc train")) +
  geom_point(aes(y = dokladnosc_test, colour = "Dokladnosc test")) +
  geom_point(aes(y = dokladnosc_rpart, colour = "Dokladnosc rpart")) + 
  ylim(0, 1) + 
  ylab("Dokladnosc") +
  ggtitle("Wplyw type na dokladnosc w drzewach dla klasyfikacji wieloklasowej") +
  labs(colour="Dane")


#Podsumowanie metod własnych
podsumowanie_regresja <-  data.frame(matrix(ncol = 8, nrow = 1))
colnames(podsumowanie_regresja) <- c("mse_train","mse_test","mse_caret","mape_train","mape_test","mape_caret","id","method")
res_knn_reg['id'] <- row.names(res_knn_reg)
res_nn_reg['id'] <- row.names(res_nn_reg)
res_tree_reg['id'] <- row.names(res_tree_reg)
res_knn_reg['method'] <- "KNN"
res_nn_reg['method'] <- "NN"
names(res_nn_reg)[names(res_nn_reg) == "mse_neuralnet"] <- "mse_caret"
names(res_nn_reg)[names(res_nn_reg) == "mape_neuralnet"] <- "mape_caret"
res_tree_reg['method'] <- "TREE"
names(res_tree_reg)[names(res_tree_reg) == "mse_rpart"] <- "mse_caret"
names(res_tree_reg)[names(res_tree_reg) == "mape_rpart"] <- "mape_caret"
podsumowanie_regresja <- podsumowanie_regresja[-1,]
podsumowanie_regresja <- rbind(podsumowanie_regresja, res_knn_reg[,c(1:6,8,9)])
podsumowanie_regresja <- rbind(podsumowanie_regresja, res_nn_reg[,c(1:6,10,11)])
podsumowanie_regresja <- rbind(podsumowanie_regresja, res_tree_reg[,c(1:6,10,11)])
podsumowanie_regresja <- podsumowanie_regresja[order(podsumowanie_regresja$mape_test),]
print("Najlepsze z wszystkich modeli dla regresji")
head(podsumowanie_regresja,5)


podsumowanie_binarna <-  data.frame(matrix(ncol = 14, nrow = 1))
colnames(podsumowanie_binarna) <- c("dokladnosc_train","dokladnosc_test","dokladnosc_caret","czulosc_train","czulosc_test","czulosc_caret","precyzja_train","precyzja_test",'precyzja_caret','specyficznosc_train','specyficznosc_test','specyficznosc_caret',"id","method")
res_knn_bin['id'] <- row.names(res_knn_bin)
res_nn_bin['id'] <- row.names(res_nn_bin)
res_tree_bin['id'] <- row.names(res_tree_bin)
res_knn_bin['method'] <- "KNN"
res_nn_bin['method'] <- "NN"
names(res_nn_bin)[names(res_nn_bin) == "dokladnosc_neuralnet"] <- "dokladnosc_caret"
names(res_nn_bin)[names(res_nn_bin) == "czulosc_neuralnet"] <- "czulosc_caret"
names(res_nn_bin)[names(res_nn_bin) == "precyzja_neuralnet"] <- "precyzja_caret"
names(res_nn_bin)[names(res_nn_bin) == "specyficznosc_neuralnet"] <- "specyficznosc_caret"
res_tree_bin['method'] <- "TREE"
names(res_tree_bin)[names(res_tree_bin) == "dokladnosc_rpart"] <- "dokladnosc_caret"
names(res_tree_bin)[names(res_tree_bin) == "czulosc_rpart"] <- "czulosc_caret"
names(res_tree_bin)[names(res_tree_bin) == "precyzja_rpart"] <- "precyzja_caret"
names(res_tree_bin)[names(res_tree_bin) == "specyficznosc_rpart"] <- "specyficznosc_caret"
podsumowanie_binarna <- podsumowanie_binarna[-1,]
podsumowanie_binarna <- rbind(podsumowanie_binarna, res_knn_bin[,c(2:15)])
podsumowanie_binarna <- rbind(podsumowanie_binarna, res_nn_bin[,c(4:17)])
podsumowanie_binarna <- rbind(podsumowanie_binarna, res_tree_bin[,c(4:17)])
podsumowanie_binarna <- podsumowanie_binarna[order(-podsumowanie_binarna$dokladnosc_test),]
print("Najlepsze z wszystkich modeli dla klasyfikacji binarnej")
head(podsumowanie_binarna,5)

podsumowanie_wieloklasowa <-  data.frame(matrix(ncol = 14, nrow = 1))
colnames(podsumowanie_wieloklasowa) <- c("dokladnosc_train","dokladnosc_test","dokladnosc_caret","recall_train","recall_test","recall_caret","precyzja_train","precyzja_test",'precyzja_caret','auc_train','auc_test','auc_caret',"id","method")
res_knn_wiel['id'] <- row.names(res_knn_wiel)
res_nn_wiel['id'] <- row.names(res_nn_wiel)
res_tree_wiel['id'] <- row.names(res_tree_wiel)
res_knn_wiel['method'] <- "KNN"
res_nn_wiel['method'] <- "NN"
names(res_nn_wiel)[names(res_nn_wiel) == "dokladnosc_neuralnet"] <- "dokladnosc_caret"
names(res_nn_wiel)[names(res_nn_wiel) == "recall_neuralnet"] <- "recall_caret"
names(res_nn_wiel)[names(res_nn_wiel) == "precyzja_neuralnet"] <- "precyzja_caret"
names(res_nn_wiel)[names(res_nn_wiel) == "auc_neuralnet"] <- "auc_caret"
res_tree_wiel['method'] <- "TREE"
names(res_tree_wiel)[names(res_tree_wiel) == "dokladnosc_rpart"] <- "dokladnosc_caret"
names(res_tree_wiel)[names(res_tree_wiel) == "recall_rpart"] <- "recall_caret"
names(res_tree_wiel)[names(res_tree_wiel) == "precyzja_rpart"] <- "precyzja_caret"
names(res_tree_wiel)[names(res_tree_wiel) == "auc_rpart"] <- "auc_caret"
podsumowanie_wieloklasowa <- podsumowanie_wieloklasowa[-1,]
podsumowanie_wieloklasowa <- rbind(podsumowanie_wieloklasowa, res_knn_wiel[,c(2:15)])
podsumowanie_wieloklasowa <- rbind(podsumowanie_wieloklasowa, res_nn_wiel[,c(4:17)])
podsumowanie_wieloklasowa <- rbind(podsumowanie_wieloklasowa, res_tree_wiel[,c(4:17)])
podsumowanie_wieloklasowa <- podsumowanie_wieloklasowa[order(-podsumowanie_wieloklasowa$dokladnosc_test),]
print("Najlepsze z wszystkich modeli dla klasyfikacji wieloklasowej")
head(podsumowanie_wieloklasowa,5)

# podsumowanie metod wbudowanych
podsumowanie_regresja_2 <- podsumowanie_regresja[order(podsumowanie_regresja$mape_caret),]
print("Najlepsze z wszystkich modeli wbudowanych dla regresji")
head(podsumowanie_regresja_2,5)

podsumowanie_binarna_2 <- podsumowanie_binarna[order(-podsumowanie_binarna$dokladnosc_caret),]
print("Najlepsze z wszystkich modeli wbudowanych dla klasyfikacji binarnej")
head(podsumowanie_binarna_2,5)

podsumowanie_wieloklasowa_2 <- podsumowanie_wieloklasowa[order(-podsumowanie_wieloklasowa$dokladnosc_caret),]
print("Najlepsze z wszystkich modeli wbudowanych dla klasyfikacji wieloklasowej")
head(podsumowanie_wieloklasowa_2,5)