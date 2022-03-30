#Funkcja do zmiany zmiennych porzadkowych
encode_ordinal <- function(x, order) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}
#Normalizacja
MinMax <- function( x, new_min = 0, new_max = 1 ){
  return( ( ( x - min(x) ) / ( max(x) - min(x) ) ) * ( new_max - new_min ) + new_min )
}
#Bledy
MAE <- function(y_tar, y_hat){
  return(mean(abs(y_tar - y_hat)))
}

MSE <- function( y_tar, y_hat ){
  return( mean( ( y_tar - y_hat )^2 ) )
}

MAPE <- function(y_tar, y_hat){
  res <- abs((y_tar - y_hat)/y_tar)
  res[!is.finite(as.numeric(unlist(res)))] <- 0
  return (mean(unlist(res)))
}

#odleglosc euklidesowa
d_euklides <- function(x, y){
  return(sqrt(sum((x - y)^2)))
}
 #dokladnosc precyzja czulosc specyficznosc
precyzja <- function(y_tar, y_hat)
{
  return(sum(y_tar == 1 & y_hat == 1)/sum(y_hat == 1))
}
czulosc <- function(y_tar, y_hat)
{
  return( sum(y_tar == 1 & y_hat == 1) / sum(y_tar == 1))
}
specyficznosc <- function(y_tar, y_hat)
{
  return( sum(y_tar == 0 & y_hat == 0) / sum(y_tar == 0))
}
dokladnosc <- function(y_tar, y_hat){
  return((sum(y_tar == 0 & y_hat == 0) + sum(y_tar == 1 & y_hat == 1)) / (sum(y_tar == 1) + sum(y_tar == 0)))
}
#funkcje potrzebne do sieci
sigmoid <- function(x)
{
  return(1/(1 + exp(-x)))
}
d_sigmoid <- function( z ){
  return( z * ( 1 - z ) )
}
# typy do drzew
Gini <- function(Y) {
  return(1 - sum((unname(table(Y)) / length(Y)) ^ 2))
}
Entropy <- function(Y) {
  prob <- unname(table(Y)) / length(Y)
  result <- prob * log2(prob)
  result[prob == 0] <- 0
  result <- -sum(result)
  return(result)
}
SS <- function(Y) {
  Y <- as.numeric(Y)
  return(sum((Y - mean(Y)) ^ 2))
}

# KNN - regresja #
KNNTrain_reg <- function(X, y_tar, k) {
  if (!all(!is.na(X)) | !all(!is.na(y_tar))){
    print("Braki w danych!")
  }
  else if (k <= 0){
    print("Nieprawidlowe k! Musi byc wieksze od 0!")
  }
  else{
    if (!is.data.frame(X)){
      print("X nie jest ani ramka danych!")
      X <- data.frame(X)
      print('X zostal przeksztalcony do ramki danych')
    }
  result <- list()
  result$X <- X
  result$y <- y_tar
  result$k <- k
  return(result)
  }
}

KNNpred_reg <- function(KNNmodel, X) {
  if (!all(!is.na(X))){
    print("Braki w danych!")
  }
  odleglosc <- matrix( 0, nrow( X ), nrow( X ))
  for( i in 1:nrow( X ) ){
    for( j in 1:nrow( X )){
      odleglosc[i,j] <- d_euklides( KNNmodel$X[i,], X[j,] )
    }
  }
  pred <- double( nrow( X ) )
  for( i in 1:nrow( X ) ){
    knn <- order( odleglosc[,i] )
    knn <- knn[1:KNNmodel$k]
    y_hat <- mean( KNNmodel$y[ knn, ] )
    pred[ i ] <- y_hat
  }
  return( pred )
}
# Crossvalidacja zmiennych
CrossValidation_knn_reg <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 7, nrow = 1))
  colnames(results) <- c("mse_train","mse_test","mse_caret","mape_train","mape_test","mape_caret","k")
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                     size = (1-1/kFold)*nrow(dane),
                     replace = F )
  }
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla knn k=", k))
    
    df_results <- data.frame(matrix(ncol = 7, nrow = 1))
    colnames(df_results) <- c("mse_train","mse_test","mse_caret","mape_train","mape_test","mape_caret","k")
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      #modele test train
      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      KNNmodel <- KNNTrain_reg(X_train, y_train, k)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      predictions_train <- KNNpred_reg(KNNmodel, X_train)
      predictions_test <- KNNpred_reg(KNNmodel, X_test)
      y_train <- as.double(unlist(y_train))
      y_test <- as.double(unlist(y_test))
      #model z pakietu caret
      KNN_caret <- knnreg( X_train, y_train, k = k )
      predictions_caret <- predict( KNN_caret, X_test )
      #bledy
      mse_train <- MSE( y_train, predictions_train )
      mse_test <- MSE( y_test, predictions_test )
      mape_train <- MAPE( y_train, round(predictions_train) )
      mape_test <- MAPE( y_test, predictions_test )
      mse_caret <- MSE( y_test, predictions_caret)
      mape_caret <- MAPE( y_test, predictions_caret)
      
      df_results <- rbind(df_results, c(mse_train,mse_test,mse_caret,mape_train,mape_test,mape_caret,k))
    }
    
      df_results <- df_results[-1,]
      df_results<-colMeans(df_results)
      results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
  }

# KNN - binarne #
KNNTrain_bin <- function(X, y_tar, k,  XminNew, XmaxNew) {
  if (!all(!is.na(X)) | !all(!is.na(y_tar))){
    print("Braki w danych!")
  }
  else if (k <= 0){
    print("Nieprawidlowe k! Musi byc wieksze od 0!")
  }
  else{
    if (!is.data.frame(X)){
      print("X nie jest ani ramka danych!")
      X <- data.frame(X)
      print('X zostal przeksztalcony do ramki danych')
    }
    result <- list()
    result$X <- X
    result$y <- y_tar
    result$k <- k
    return(result)
  }
}
KNNpred_bin <- function(KNNmodel, X) {
  if (!all(!is.na(X))){
    print("Braki w danych!")
  }
  odleglosc <- matrix( 0, nrow( X ), nrow( X ))
  for( i in 1:nrow( X ) ){
    for( j in 1:nrow( X )){
      odleglosc[i,j] <- d_euklides( KNNmodel$X[i,], X[j,] )
    }
  }
  pred <- data.frame(matrix(0, nrow = nrow(X), ncol =length(levels(KNNmodel$y)) + 1))
  colnames(pred) <- c(levels(KNNmodel$y), "klasa")
  for( i in 1:nrow( X ) ){
    knn <- order( odleglosc[,i] )
    knn <- knn[1:KNNmodel$k]
    y_hat <- summary(KNNmodel$y[knn]) / length(levels(KNNmodel$y))
    klasa <- names(y_hat[y_hat == max(y_hat)][1])
    pred[i,] <- c(y_hat, klasa)
  }
  return( pred )
}
# Crossvalidacja zmiennych
CrossValidation_knn_bin <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 13, nrow = 1))
  colnames(results) <- c("k","dokladnosc_train","dokladnosc_test","dokladnosc_caret","czulosc_train","czulosc_test","czulosc_caret","precyzja_train","precyzja_test",'precyzja_caret','specyficznosc_train','specyficznosc_test','specyficznosc_caret')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla knn k=", k))
    
    df_results <- data.frame(matrix(ncol =13, nrow = 1))
    colnames(df_results) <- c("k","dokladnosc_train","dokladnosc_test","dokladnosc_caret","czulosc_train","czulosc_test","czulosc_caret","precyzja_train","precyzja_test",'precyzja_caret','specyficznosc_train','specyficznosc_test','specyficznosc_caret')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]

      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      y_train_factor <- factor(unlist(y_train))
      KNNmodel <- KNNTrain_bin(X_train, y_train_factor, k)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test_factor <- factor(unlist(y_test))
      #model train
      predictions_train <- KNNpred_bin(KNNmodel, X_train)
      predictions_train_factor <- factor(predictions_train$klasa)
      #miary dokladnosci modelu
      dokladnosc_train <- dokladnosc(y_train_factor, predictions_train_factor)
      czulosc_train <- czulosc(y_train_factor, predictions_train_factor)
      precyzja_train <- precyzja(y_train_factor, predictions_train_factor)
      specyficznosc_train <- specyficznosc(y_train_factor, predictions_train_factor)
      #model test
      predictions_test <- KNNpred_bin(KNNmodel, X_test)
      predictions_test_factor <- factor(predictions_test$klasa)
      #miary dokladnosci modelu
      dokladnosc_test <- dokladnosc(y_test_factor, predictions_test_factor)
      czulosc_test <- czulosc(y_test_factor, predictions_test_factor)
      precyzja_test <- precyzja(y_test_factor, predictions_test_factor)
      specyficznosc_test <- specyficznosc(y_test_factor, predictions_test_factor)
      #modele wbudowane z biblioteki caret
      KNN_caret <- (knnreg( X_train, as.numeric(unlist(y_train)), k = k ))
      predictions_caret <- round(predict( KNN_caret, X_test )-1)
      predictions_caret_factor <- as.factor(predictions_caret)
      #miary dokladnosci modelu
      dokladnosc_caret <- dokladnosc(y_test_factor, predictions_caret_factor)
      czulosc_caret <- czulosc(y_test_factor, predictions_caret_factor)
      precyzja_caret <- precyzja(y_test_factor, predictions_caret_factor)
      specyficznosc_caret <- specyficznosc(y_test_factor, predictions_caret_factor)
      
      df_results <- rbind(df_results, c(k, dokladnosc_train,dokladnosc_test,dokladnosc_caret,czulosc_train,czulosc_test,czulosc_caret,precyzja_train,precyzja_test,precyzja_caret,specyficznosc_train,specyficznosc_test,specyficznosc_caret))
    }
    
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}

# KNN - wieloklasowa #
KNNTrain_wiel <- function(X, y_tar, k,  XminNew, XmaxNew) {
  if (!all(!is.na(X)) | !all(!is.na(y_tar))){
    print("Braki w danych!")
  }
  else if (k <= 0){
    print("Nieprawidlowe k! Musi byc wieksze od 0!")
  }
  else{
    if (!is.data.frame(X)){
      print("X nie jest ani ramka danych!")
      X <- data.frame(X)
      print('X zostal przeksztalcony do ramki danych')
    }
    result <- list()
    result$X <- X
    result$y <- y_tar
    result$k <- k
    return(result)
  }
}
KNNpred_wiel <- function(KNNmodel, X) {
  if (!all(!is.na(X))){
    print("Braki w danych!")
  }
  odleglosc <- matrix( 0, nrow( X ), nrow( X ))
  for( i in 1:nrow( X ) ){
    for( j in 1:nrow( X )){
      odleglosc[i,j] <- d_euklides( KNNmodel$X[i,], X[j,] )
    }
  }
  pred <- data.frame(matrix(0, nrow = nrow(X), ncol =length(levels(KNNmodel$y)) + 1))
  colnames(pred) <- c(levels(KNNmodel$y), "klasa")
  for( i in 1:nrow( X ) ){
    knn <- order( odleglosc[,i] )
    knn <- knn[1:KNNmodel$k]
    y_hat <- summary(KNNmodel$y[knn]) / length(levels(KNNmodel$y))
    klasa <- names(y_hat[y_hat == max(y_hat)][1])
    pred[i,] <- c(y_hat, klasa)
  }
  return( pred )
}
# Crossvalidacja zmiennych
CrossValidation_knn_wiel <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 13, nrow = 1))
  colnames(results) <- c("k","dokladnosc_train","dokladnosc_test","dokladnosc_caret","recall_train","recall_test","recall_caret","precyzja_train","precyzja_test",'precyzja_caret','auc_train','auc_test','auc_caret')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla knn k=", k))
    
    df_results <- data.frame(matrix(ncol =13, nrow = 1))
    colnames(df_results) <- c("k","dokladnosc_train","dokladnosc_test","dokladnosc_caret","recall_train","recall_test","recall_caret","precyzja_train","precyzja_test",'precyzja_caret','auc_train','auc_test','auc_caret')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]

      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      y_train_factor <- factor(unlist(y_train),levels=c("L", "B", "R"))
      KNNmodel <- KNNTrain_wiel(X_train, y_train_factor, k)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test_factor <- factor(unlist(y_test),levels=c("L", "B", "R"))
      y_train_numeric <- ifelse( y_train == "L", 1, ifelse( y_train == "B", 2, 3) )
      y_test_numeric<-ifelse( y_test == "L", 1, ifelse( y_test == "B", 2, 3) )
      #model train
      predictions_train <- KNNpred_wiel(KNNmodel, X_train)
      predictions_train_factor <- factor(predictions_train$klasa, levels=c("L", "B", "R"))
      predictions_train <-  ifelse( predictions_train_factor == "L", 1, ifelse( predictions_train_factor == "B", 2, 3))
      results_train <- confusionMatrix( y_train_factor, predictions_train_factor )$table
      #miary dokladnosci modelu (klasyfikacja wieloklasowa result - macierz 3x3)
      dokladnosc_train <- sum(diag(results_train)) / sum(results_train)
      recall_train <- sum(diag(results_train)) / sum(apply(results_train, 1, sum))
      precyzja_train <- sum(diag(results_train)) / sum(apply(results_train, 2, sum))
      auc_train <- auc(multiclass.roc(y_train_numeric, predictions_train, levels=c(1,2,3)))[1]
      #model test
      predictions_test <- KNNpred_wiel(KNNmodel, X_test)
      predictions_test_factor <- factor(predictions_test$klasa, levels=c("L", "B", "R"))
      predictions_test <-  ifelse( predictions_test_factor == "L", 1, ifelse( predictions_train_factor == "B", 2, 3))
      results_test <- confusionMatrix( y_test_factor, predictions_test_factor )$table
      #miary dokladnosci modelu
      dokladnosc_test <- sum(diag(results_test)) / sum(results_test)
      recall_test <- sum(diag(results_test)) / sum(apply(results_test, 1, sum))
      precyzja_test <- sum(diag(results_test)) / sum(apply(results_test, 2, sum))
      auc_test <- auc(multiclass.roc(y_test_numeric, predictions_test, levels=c(1,2,3)))[1]
      #model wbudowany z biblioteki caret
      KNN_caret <- knnreg( X_train, y_train_numeric, k = k )
      predictions_caret <- round(predict( KNN_caret, X_test ))
      predictions_caret_factor <- factor(predictions_caret, levels = c(1,2,3))
      results_caret <- confusionMatrix( factor(y_test_numeric, levels=c(1,2,3)), predictions_caret_factor )$table
      #miary dokladnosci modelu
      dokladnosc_caret <- sum(diag(results_caret)) / sum(results_caret)
      recall_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 1, sum))
      precyzja_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 2, sum))
      auc_caret <- auc(multiclass.roc(y_test_factor, predictions_caret))[1]
      
      df_results <- rbind(df_results, c(k, dokladnosc_train,dokladnosc_test,dokladnosc_caret,recall_train,recall_test,recall_caret,precyzja_train,precyzja_test,precyzja_caret,auc_train,auc_test,auc_caret))
    }
    
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}

# Sieci - regresja #
linear <- function (z){
  return(z)
}
Wprzod_reg <- function(X, W) {
  H <- list()
  H[[1]] <- as.matrix(X) %*% W[[1]]
  X <-as.list(X)
  for(i in 2:length(W)) {
    if(i==length(W)){
      H[[i]] <- H[[i - 1]] %*% W[[i]]
    }
    else{
      H[[i]] <- sigmoid(H[[i - 1]] %*% W[[i]])
    }
  }
  return(H)
}

Wstecz_reg <- function(X, Y, W, H, lr) {
  y_hat <- H[[length(H)]]
  dW <- list()
  dH <- list()
  dH[[length(W)]] <- Y - y_hat
  
  for(i in (length(W)):1) {
    if(i != length(W)) {
      dH[[i]] <- as.matrix(dH[[i + 1]]) %*% t(W[[i + 1]]) * d_sigmoid(H[[i]])
    }
    if(i != 1) {
      dW[[i]] <- t(H[[i - 1]]) %*% as.matrix(dH[[i]])
    }
  }
  dW[[1]] <- t(X) %*% dH[[1]]
  
  for(i in 1:length(dW)) {
    W[[i]] <- W[[i]] + lr * dW[[i]]
  }
  return(W)
}

trainNN_reg <- function(X, Y, h, lr=0.01, iter=1000, seed=111){
  set.seed(seed)
  W <- list()
  n_x <- length(colnames(X))
  H_sizes <- c(n_x, h, 1)
  
  for(i in 1:(length(H_sizes) - 1)) {
    W[[i]] <- matrix(runif(H_sizes[i] * H_sizes[i + 1], -1, 1 ), nrow = H_sizes[i])
  }
  for(i in 1:iter) {
    H <- Wprzod_reg((X),W)
    W <- Wstecz_reg(X, Y, W, H, lr)
  }
  return(list(y_hat = H[[length(H)]], W = W))
}

predNN_reg <- function( Xnew, NN ){
  len <- length(NN$W)
  for (i in 1:len){
    if(i == 1){
      z <- as.matrix(Xnew)  %*% NN$W[[i]]
      H <- sigmoid( z )
    }
    else if(i == len){
      z <- H  %*% NN$W[[i]]
      y_hat <- linear(z)
    }
    else{
      z <- H  %*% NN$W[[i]]
      H <- sigmoid( z )
    }
  }
  return(y_hat)
}

# Crossvalidacja zmiennych
CrossValidation_nn_reg <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 9, nrow = 1))
  colnames(results) <- c("mse_train","mse_test","mse_neuralnet","mape_train","mape_test","mape_neuralnet","hn","lr","iter")
  set.seed(seed)
  n <- nrow(data)
  indxT <- list()
  for(i in 1:kFold){
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  for (i in 1:nrow(parTune)){
    print(paste0("Obliczam wyniki dla sieci ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla sieci z parametrami nr: ", i))
    print(1L)
    h <- parTune[i,]$h[[1]]
    lr <- parTune[i,]$lr
    iter <- parTune[i,]$iter
    df_results <- data.frame(matrix(ncol = 9, nrow = 1))
    colnames(df_results) <- c("mse_train","mse_test","mse_neuralnet","mape_train","mape_test","mape_neuralnet","hn","lr","iter")
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      
      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      y_train <- as.data.frame(y_train)

      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test <- as.data.frame(y_test)

      NN_reg <- trainNN_reg(X_train, y_train, h = h, lr = lr, iter = iter,seed=seed)
      #modele test i train
      predictions_train <- predNN_reg(X_train, NN_reg)
      predictions_test <- predNN_reg(X_test, NN_reg)
      #bledy
      mse_train <- MSE( y_train[["Y"]], predictions_train )
      mse_test <- MSE( y_test[["Y"]], predictions_test )
      y_train <- as.numeric(unlist(y_train))
      mape_train <- MAPE( y_train, predictions_train )
      y_test <- as.numeric(unlist(y_test))
      mape_test <- MAPE( y_test, predictions_test )
      
      #model z biblioteki neuralnet
      NN_caret <- neuralnet(Y ~ Point_X+ Point_Y+ month+ day+ FFMC+ DMC+ DC+ ISI+ temp+ RH+ wind, data_train, linear.output = FALSE)	
      predictions_caret <- NN_caret$net.result[[1]]
      #bledy
      mse_caret <- MSE( y_train, predictions_caret)
      mape_caret <- MAPE( y_train, predictions_caret)
      
      df_results <- rbind(df_results, c(mse_train,mse_test,mse_caret,mape_train,mape_test,mape_caret,0,0,0))
      
    }
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['hn']] <- paste(h, collapse = " ")
    df_results[['lr']] = lr
    df_results[['iter']] = iter
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}

#Sieci - binarna#
Wprzod_bin <- function(X, W) {
  H <- list()
  H[[1]] <- as.matrix(X) %*% W[[1]]
  X <-as.list(X)
  for(i in 2:length(W)) {
    H[[i]] <- sigmoid(H[[i - 1]] %*% W[[i]])
  }
  return(H)
}

Wstecz_bin <- function(X, Y, W, H, lr) {
  y_hat <- H[[length(H)]]
  dW <- list()
  dH <- list()
  dH[[length(W)]] <- (Y - y_hat) * d_sigmoid(y_hat)
  
  for(i in (length(W)):1) {
    if(i != length(W)) {
      dH[[i]] <- as.matrix(dH[[i + 1]]) %*% t(W[[i + 1]]) * d_sigmoid(H[[i]]) 
    }
    if(i != 1) {
      dW[[i]] <- t(H[[i - 1]]) %*% as.matrix(dH[[i]])
    }
  }
  dW[[1]] <- t(X) %*% dH[[1]]
  
  for(i in 1:length(dW)) {
    W[[i]] <- W[[i]] + lr * dW[[i]]
  }
  return(W)
}

trainNN_bin <- function(X,Y, h, lr, iter) {
  W <- list()
  n_x <- length(colnames(X))
  H_sizes <- c(n_x, h, 1)
  
  for(i in 1:(length(H_sizes) - 1)) {
    W[[i]] <- matrix(runif(H_sizes[i] * H_sizes[i + 1], -1, 1 ), nrow = H_sizes[i])
  }
  for(i in 1:iter) {
    H <- Wprzod_bin(X,W)
    W <- Wstecz_bin(X, Y, W, H, lr)
  }
  result <- length(H)
  return(list(y_hat = H[[result]], W = W))
}

predNN_bin <- function( Xnew, NN ){
  H <- list()
  H[[1]] <- as.matrix(Xnew) %*% NN[[1]]
  
  for(i in 2:length(NN)) {
    H[[i]] <- sigmoid(H[[i - 1]] %*% NN[[i]])
  }
  res <- length(H)
  return(H[res])
}

CrossValidation_nn_bin <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 15, nrow = 1))
  colnames(results) <- c("h","lr","iter","dokladnosc_train","dokladnosc_test","dokladnosc_neuralnet","czulosc_train","czulosc_test","czulosc_neuralnet","precyzja_train","precyzja_test",'precyzja_neuralnet','specyficznosc_train','specyficznosc_test','specyficznosc_neuralnet')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  
  for (i in 1:nrow(parTune)){
    print(paste0("Obliczam wyniki dla sieci ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla sieci z parametrami nr: ", i))
    print(1L)
    h <- parTune[i,]$h[[1]]
    lr <- parTune[i,]$lr
    iter <- parTune[i,]$iter
    df_results <- data.frame(matrix(ncol =15, nrow = 1))
    colnames(df_results) <- c("h","lr","iter","dokladnosc_train","dokladnosc_test","dokladnosc_neuralnet","czulosc_train","czulosc_test","czulosc_neuralnet","precyzja_train","precyzja_test",'precyzja_neuralnet','specyficznosc_train','specyficznosc_test','specyficznosc_neuralnet')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      X_train <- subset(data_train, select = -c(Y))
      y_train <- (subset(data_train, select = c(Y)))
      y_train_factor <- matrix(as.numeric(factor(unlist(y_train),levels=c(0,1)))-1)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test_factor <- matrix(as.numeric(factor(unlist(y_test),levels=c(0,1)))-1)
      
      NN_bin <- trainNN_bin( X_train,y_train_factor, h = h, lr = lr, iter = iter)
      #model train
      predictions_train <- predNN_bin(X_train, NN_bin$W)
      predictions_train <- predictions_train[[1]]
      predictions_train <- ifelse(predictions_train[,1] >= 0.5, 1, 0)
      predictions_train_factor <- factor(predictions_train, levels=c(0,1))
      #miary dokladnosci dla train
      dokladnosc_train <- dokladnosc(y_train_factor, predictions_train_factor)
      czulosc_train <- czulosc(y_train_factor, predictions_train_factor)
      results_train <- confusionMatrix( factor(unlist(y_train),levels=c(0,1)), predictions_train_factor )$table
      precyzja_train <- sum(diag(results_train)) / sum(apply(results_train, 2, sum))
      specyficznosc_train <- specyficznosc(y_train_factor, predictions_train_factor)
      #model test      
      predictions_test <- predNN_bin(X_test, NN_bin$W)
      predictions_test <- predictions_test[[1]]
      predictions_test <- ifelse(predictions_test[,1] >= 0.5, 1, 0)
      predictions_test_factor <- factor(predictions_test, levels=c(0,1))
      #miary dokladnosci test
      dokladnosc_test <- dokladnosc(y_test_factor, predictions_test_factor)
      czulosc_test <- czulosc(y_test_factor, predictions_test_factor)
      results_test <- confusionMatrix( factor(unlist(y_test),levels=c(0,1)), predictions_test_factor)$table
      precyzja_test <- sum(diag(results_test)) / sum(apply(results_test, 2, sum))
      specyficznosc_test <- specyficznosc(y_test_factor, predictions_test_factor)
      #model z biblioteki neuralnet
      NN_caret <- neuralnet(Y ~ IR+ MR+ FF+ CR+ CO+ OP, data_train,linear.output = FALSE, hidden=h)
      predictions_caret <- round((NN_caret$net.result[[1]]))
      predictions_caret_factor <- factor(predictions_caret, levels=c(0,1))
      #miary dokladnosci modelu
      dokladnosc_caret <- dokladnosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      czulosc_caret <- czulosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      precyzja_caret <- precyzja(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      specyficznosc_caret <- specyficznosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      
      df_results <- rbind(df_results, c(0,0,0, dokladnosc_train,dokladnosc_test,dokladnosc_caret,czulosc_train,czulosc_test,czulosc_caret,precyzja_train,precyzja_test,precyzja_caret,specyficznosc_train,specyficznosc_test,specyficznosc_caret))
    }
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['h']] <- paste(h, collapse = " ")
    df_results[['lr']] = lr
    df_results[['iter']] = iter
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}

#Sieci - wieloklasowa#
SoftMax <- function(z){
  return(exp(z) / sum(exp(z)))}

Wprzod_wiel <- function(X, W) {
  H <- list()
  H[[1]] <- as.matrix(X) %*% W[[1]]
  X <-as.list(X)
  for(i in 2:length(W)) {
    if(i==length(W)){
      H[[i]] <- t(apply( H[[i - 1]] %*% W[[i]], 1, SoftMax ))
    }
    else{
      H[[i]] <- sigmoid(H[[i - 1]] %*% W[[i]])
    }
  }
  return(H)
}

Wstecz_wiel <- function(X, Y, W, H, lr) {
  y_hat <- H[[length(H)]]
  dW <- list()
  dH <- list()
  dH[[length(W)]] <- (Y - y_hat) / nrow( X )
  
  for(i in (length(W)):1) {
    if(i != length(W)) {
      dH[[i]] <- as.matrix(dH[[i + 1]]) %*% t(W[[i + 1]]) * d_sigmoid(H[[i]])
    }
    if(i != 1) {
      dW[[i]] <- t(H[[i - 1]]) %*% (dH[[i]])
    }
  }
  dW[[1]] <- t(X) %*% dH[[1]]
  
  for(i in 1:length(dW)) {
    W[[i]] <- W[[i]] + lr * dW[[i]]
  }
  return(W)
}

trainNN_wiel <- function(X,Y, h, lr, iter) {
  W <- list()
  n_x <- ncol(X)
  H_sizes <- c(n_x, h, ncol(Y))
  
  for(i in 1:(length(H_sizes) - 1)) {
    W[[i]] <- matrix(runif(H_sizes[i] * H_sizes[i + 1], -1, 1 ), nrow = H_sizes[i])
  }
  for(i in 1:iter) {
    H <- Wprzod_wiel(X,W)
    W <- Wstecz_wiel(X, Y, W, H, lr)
  }
  return(list(y_hat = H[[length(H)]], W = W))
}

predNN_wiel <- function( Xnew, W ){
  H <- list()
  H[[1]] <- as.matrix(Xnew) %*% W[[1]]
  
  for(i in 2:length(W)) {
    if(i==length(W)){
      H[[i]] <- matrix( t( apply( H[[i - 1]] %*% W[[i]], 1, SoftMax ) ), nrow = nrow(Xnew) )
    }
    else{
      H[[i]] <- sigmoid(H[[i - 1]] %*% W[[i]])
    }
  }
  res <- length(H)
  return(H[res])
}

CrossValidation_nn_wiel <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 15, nrow = 1))
  colnames(results) <- c("h","lr","iter","dokladnosc_train","dokladnosc_test","dokladnosc_neuralnet","recall_train","recall_test","recall_neuralnet","precyzja_train","precyzja_test",'precyzja_neuralnet','auc_train','auc_test','auc_neuralnet')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  
  for (i in 1:nrow(parTune)){
    print(paste0("Obliczam wyniki dla sieci ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla sieci z parametrami nr: ", i))
    print(1L)
    h <- parTune[i,]$h[[1]]
    lr <- parTune[i,]$lr
    iter <- parTune[i,]$iter
    df_results <- data.frame(matrix(ncol =15, nrow = 1))
    colnames(df_results) <- c("h","lr","iter","dokladnosc_train","dokladnosc_test","dokladnosc_neuralnet","recall_train","recall_test","recall_neuralnet","precyzja_train","precyzja_test",'precyzja_neuralnet','auc_train','auc_test','auc_neuralnet')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      
      X_train <- subset(data_train, select = -c(Y))
      y_train <- model.matrix( ~ Y - 1, data_train )
      y_train_numeric <- apply(X=y_train, MARGIN = 1, FUN=which.max)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- model.matrix( ~ Y - 1, data_test )
      y_test_numeric <- apply(X=y_test, MARGIN = 1, FUN=which.max)
      
      NN_wiel <- trainNN_wiel( X_train,y_train, h = h, lr = lr, iter = iter)
      #model train
      predictions_train <- predNN_wiel(X_train, NN_wiel$W)
      predictions_train <- apply(X=predictions_train[[1]], MARGIN = 1, FUN=which.max)
      predictions_train_factor <- factor(predictions_train, levels=c(1,2,3))
      results_train <- confusionMatrix(factor(y_train_numeric), predictions_train_factor)$table
      #miary dokladnosci train
      dokladnosc_train <- sum(diag(results_train)) / sum(results_train)
      recall_train <- sum(diag(results_train)) / sum(apply(results_train, 1, sum))
      precyzja_train <- sum(diag(results_train)) / sum(apply(results_train, 2, sum))
      auc_train <- auc(multiclass.roc(y_train_numeric, predictions_train, levels=c(1,2,3)))[1]
      #model test      
      predictions_test <- predNN_wiel(X_test, NN_wiel$W)
      predictions_test <- apply(X=predictions_test[[1]], MARGIN = 1, FUN=which.max)
      predictions_test_factor <- factor(predictions_test, levels=c(1,2,3))
      #miary dokladnosci test
      results_test <- confusionMatrix(factor(y_test_numeric), predictions_test_factor)$table
      dokladnosc_test <- sum(diag(results_test)) / sum(results_test)
      recall_test <- sum(diag(results_test)) / sum(apply(results_test, 1, sum))
      precyzja_test <- sum(diag(results_test)) / sum(apply(results_test, 2, sum))
      auc_test <- auc(multiclass.roc(y_test_numeric, predictions_test, levels=c(1,2,3)))[1]
      #model z biblioteki neuralnet
      NN_caret <- neuralnet(Y ~ X1 + X1.1 + X1.2 + X1.3, data_train, linear.output = FALSE, hidden = h)
      predictions_caret <- round((NN_caret$net.result[[1]]))
      predictions_caret <- apply(X=predictions_caret, MARGIN = 1, FUN=which.max)
      predictions_caret_factor <- factor(predictions_caret, levels=c(1,2,3))
      results_caret <- confusionMatrix(factor(y_train_numeric), predictions_caret_factor)$table
      #miary dokladnosci neuralnet
      dokladnosc_caret <- sum(diag(results_caret)) / sum(results_caret)
      recall_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 1, sum))
      precyzja_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 2, sum))
      auc_caret <- auc(multiclass.roc(y_train_numeric, predictions_caret, levels=c(1,2,3)))[1]
      
      df_results <- rbind(df_results, c(0,0,0, dokladnosc_train,dokladnosc_test,dokladnosc_caret,recall_train,recall_test,recall_caret,precyzja_train,precyzja_test,precyzja_caret,auc_train,auc_test,auc_caret))
    }
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['h']] <- paste(h, collapse = " ")
    df_results[['lr']] = lr
    df_results[['iter']] = iter
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  
  return(results)
}
# DRZEWA #
library( data.tree )
StopIfNot <- function(Y, X, data, type, depth, minobs = 2, overfit, cf) {
  if (!is.data.frame(data)) {
    return("Dane nie sa ramka danych!")
  }
  if (!all(all(X %in% names(data)), all(Y %in% names(data)))) {
    print("Sa wartosci w X lub Y ktore nie istnieja w nazwach kolumn w danych")
    return(FALSE)
  }
  if (any(is.na(X)) & any(is.na(Y))) {
    print("Wystepuja braki danych!")
    return(FALSE)
  }
  if (depth < 0) {
    print("Nieprawdidlowy argument depth!")
    return(FALSE)
  }
  if (minobs < 0) {
    print("Minobs musi byc wieksze od 0!")
    return(FALSE)
  }
  if (!any(type %in% c("Gini", "Entropy", "SS"))) {
    print("Nieprawidlowy typ!")
    return(FALSE)
  }
  if (!any(overfit %in% c("none", "prune"))) {
    print("Nieprawidlowy argument overfit!")
    return(FALSE)
  }
  if (cf<=0 || cf>0.5) {
    print("Argument cf musi byc w przedziale (0;0.5]")
    return(FALSE)
  }
  return(TRUE)
}

AssignInitialMeasures <- function(tree, Y, data, type, depth) {
  tree$depth <- 0
  giniImpurity <- function(Y) {
    return(1 - sum((unname(table(Y)) / length(Y)) ^ 2))
  }
  entropy <- function(Y) {
    prob <- unname(table(Y)) / length(Y)
    result <- prob * log2(prob)
    result[prob == 0] <- 0
    result <- -sum(result)
    return(result)
  }
  ss <- function(Y) {
    Y <- as.numeric(Y)
    return(sum((Y - mean(Y)) ^ 2))
  }
  
  if (type == "Gini") {
    tree$Gini <- giniImpurity(data[[Y]])
  } else if (type == "Entropy") {
    tree$Entropy <- entropy(data[[Y]])
  } else if (type == "SS") {
    tree$SS <- ss(data[[Y]])
  }
  
  return(tree)
}
AssignInfo <- function(tree, Y, X, data, type, depth, minobs, overfit, cf) {
  tree$Y <- Y
  tree$X <- X
  tree$data <- data
  tree$type <- type
  tree$depth <- depth
  tree$minobs <- minobs
  tree$overfit <- overfit
  tree$cf <- cf
  return(tree)
}
FindBestSplit <- function(Y, X, data, parentVal, type, minobs) {
  SplitVar <- function(Y, X, parentVal, type, minobs) {
    nominal_scale <- is.factor(X) & !is.ordered(X)
    ordinal_scale <- is.factor(X) & is.ordered(X)
    if(nominal_scale) {
      splits <- numeric(0)
      for (level in levels(X)) {
        splits[level] <- mean(as.numeric(Y[X == level]))
      }
      splits <- sort(splits)
    } 
    else {
      s <- unique(X)
      if (length(X) == 1) {
        splits <- s
      } else{
        splits <- head(sort(s), -1)
      }
    }
    n <- length(X)
    result <- data.frame(matrix(0, length(splits), 6))
    colnames(result) <- c("InfGain", "lVal", "rVal", "point", "ln", "rn")
    for (i in 1:length(splits)) {
      if(nominal_scale) {
        partition <- X %in% names(splits)[1:i]
      } else {
        partition <- X <= splits[i]
      }
      ln <- sum(partition)
      rn <- n - ln
      lVal <- do.call(type, list(Y[partition]))
      rVal <- do.call(type, list(Y[!partition]))
      InfGain <- parentVal - (ln / n * lVal + rn / n * rVal)
      result[i, "InfGain"] <- InfGain
      result[i, "lVal"] <- lVal
      result[i, "rVal"] <- rVal
      if(nominal_scale) {
        result[i, "point"] <- paste(names(splits)[1:i], collapse = ",")
      } 
      else if(ordinal_scale) {
        result[i, "point"] <- paste(splits[splits <= splits[i]], collapse = ",")
      } 
      else {
        result[i, "point"] <- splits[i]
      }
      result[i, "ln"] <- ln
      result[i, "rn"] <- rn
    }
    incl <- result$ln >= minobs & result$rn >= minobs & result$InfGain > 0
    result <- result[incl, , drop=F]
    best <- which.max( result$InfGain )
    
    result <- result[ best, , drop=F]
    return(result)
  }
  best_splits <- sapply( X, function( i ){
    SplitVar( Y = data[,Y] , X = data[,i], parentVal = parentVal, type = type, minobs = minobs )
  }, simplify=F)
  
  best_splits <- do.call("rbind", best_splits)
  best_split <- which.max( best_splits$InfGain )
  best_split <- best_splits[ best_split, ]
  return(best_split)
}
Tree <- function(Y, X, data, type, depth, minobs, overfit, cf) {
  if(!StopIfNot(Y, X, data, type, depth, minobs, overfit, cf)) {
    return()
  }
  tree <- Node$new( "Root" )
  AssignInitialMeasures(tree, Y, data, type, depth)
  BuildTree_reg(tree, Y = Y, Xnames = X, data = data, depth = depth, minobs = minobs, type = type, overfit = overfit, cf = cf)
  PruneTree <- function(){}
  AssignInfo(tree, Y, X, data, type, depth, minobs, overfit, cf)
  return(tree)
}
PredictTree <- function(tree, data) {
  PredictVal <- function(tree, data, firstIter = T) {
    if(!is.null(tree$Leaf)){
      if (is.null(tree$Class)) {
        return(data.frame(tree$Value))
      } else {
        return(data.frame(tree$Prob, tree$Class))
      }
    }
    nodeName <- tree$children[[1]]$name
    args <- unlist(strsplit(nodeName, " "))
    var <- args[1]
    if(args[2] == "is") {
      set <- unlist(strsplit(args[3], ","))
      node <- tree$children[[ifelse(data[var][[1]] %in% set, 1, 2)]]
    } else {
      val <- args[3]
      node <- tree$children[[ifelse( data[var] <= as.numeric(val), 1, 2)]]
    }
    PredictVal(node, data, F)
  }
  n <- nrow(data)
  results <- as.data.frame(matrix(0, n, length(levels(tree$data[[tree$Y]])) + 1))
  for(i in 1:n){
    predict_tree <- PredictVal(tree, data[i, , drop = F])
    if(length(predict_tree) == 1){
      results[i,] <- predict_tree
    }
    else{
      results[i, ] <- c(t(predict_tree[, 2]), predict_tree[1, 3])
    }
  }
  return(results)
}

#Regresja#
BuildTree_reg <- function(node, Y, Xnames, data, depth, type, minobs, overfit, cf) {
  if (type != "SS"){
    print("W regresji typ musi byc SS!!!!")
  }
  else {
    Prob <- function( y ){
      res <- unname( table( y ) )
      res <- res / sum( res )
      return( res )
    }
    node$Count <- nrow( data )
    node$Prob <- Prob( data[,Y] )
    node$Class <- levels( data[,Y] )[ which.max(node$Prob) ]
    node$Value <- mean(data[,Y])
    bestSplit <- FindBestSplit( Y, Xnames, data, parentVal =  node[[type]], minobs = minobs, type = type )
    ifStop <- nrow( bestSplit ) == 0 
    if( node$depth == depth | ifStop | all( node$Prob %in% c(0,1) ) ){
      node$Leaf <- "*"
      return( node )
    }
    if(is.character(bestSplit$point)) {
      splitIndx <- data[, rownames(bestSplit)] %in% unlist(strsplit(bestSplit$point, ","))
    } else {
      splitIndx <- data[, rownames(bestSplit)] <= bestSplit$point
    }
    childFrame <- split( data, splitIndx )
    namel <- sprintf( "%s %s %s",  rownames(bestSplit), ifelse(is.na(as.numeric(bestSplit$point)), "is", "<="), bestSplit$point)
    childL <- node$AddChild( namel )
    childL$depth <- node$depth + 1
    childL[[type]] <- bestSplit$lVal
    BuildTree_reg( node = childL, Y = Y,  Xnames = Xnames, data = childFrame[["TRUE"]], type = type, depth = depth, minobs = minobs )
    name <- sprintf( "%s %s %s",  rownames(bestSplit), ifelse(is.na(as.numeric(bestSplit$point)), "is not", ">"), bestSplit$point )
    childR <- node$AddChild( name )
    childR$depth <- node$depth + 1
    childR[[type]] <- bestSplit$rVal
    BuildTree_reg( node = childR, Y = Y,  Xnames = Xnames, data = childFrame[["FALSE"]], type = type, depth = depth, minobs = minobs )
  }
}
Tree_reg <- function(Y, X, data, type, depth, minobs, overfit, cf) {
  if(!StopIfNot(Y, X, data, type, depth, minobs, overfit, cf)) {
    return()
  }
  tree <- Node$new( "Root" )
  AssignInitialMeasures(tree, Y, data, type, depth)
  BuildTree_reg(tree, Y = Y, Xnames = X, data = data, depth = depth, minobs = minobs, type = type, overfit = overfit, cf = cf)
  PruneTree <- function(){}
  AssignInfo(tree, Y, X, data, type, depth, minobs, overfit, cf)
  return(tree)
}

CrossValidation_tree_reg <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 9, nrow = 1))
  colnames(results) <- c("mse_train","mse_test","mse_rpart","mape_train","mape_test","mape_rpart","minobs","depth",'type')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla drzew ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla drzew z parametrami nr: ", i))
    print(1L)    
    minobs <- parTune[i,]$minobs
    depth <- parTune[i,]$depth
    type <- parTune[i,]$type

    df_results <- data.frame(matrix(ncol = 9, nrow = 1))
    colnames(df_results) <- c("mse_train","mse_test","mse_rpart","mape_train","mape_test","mape_rpart","minobs","depth",'type')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      #modele test train
      tree_pred_train <- Tree_reg( Y = "Y", X = colnames(X_train), data = data_train, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_train <- PredictTree(tree_pred_train, data_train)
      tree_pred_test <- Tree_reg( Y = "Y", X = colnames(X_test), data = data_test, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_test <- PredictTree(tree_pred_test, data_train)
      #model z pakietu rpart
      cf=0.2
      tree_caret <- rpart(Y ~., data = data_train, control = rpart.control(minsplit = minobs, maxdepth = depth, cp = cf))
      predictions_caret <- predict(tree_caret, data_test)
      predictions_caret
      #bledy
      mse_train <- MSE( as.double(unlist(y_train)), as.double(unlist(predictions_train )))
      mse_test <- MSE( as.double(unlist(y_test)), as.double(unlist(predictions_test )) )
      mape_train <- MAPE( as.double(unlist(y_train)), as.double(unlist(predictions_train )) )
      mape_test <- MAPE( as.double(unlist(y_test)), as.double(unlist(predictions_test ) ))
      mse_caret <- MSE( as.double(unlist(y_test)), predictions_caret)
      mape_caret <- MAPE( as.double(unlist(y_test)), predictions_caret)
      
      df_results <- rbind(df_results, c(mse_train,mse_test,mse_caret,mape_train,mape_test,mape_caret,0,0,0))
    }
    
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['minobs']] <- minobs
    df_results[['depth']] = depth
    df_results[['type']] = type
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}


#Binarne
BuildTree_bin <- function(node, Y, Xnames, data, depth, type, minobs, overfit, cf) {
  if (type == "SS"){
    print("W klasyfikacji typ nie moze byc SS!!!!")
  }
  else {
    Prob <- function( y ){
      res <- unname( table( y ) )
      res <- res / sum( res )
      return( res )
    }
    node$Count <- nrow( data )
    node$Prob <- Prob( data[,Y] )
    node$Class <- levels( data[,Y] )[ which.max(node$Prob) ]
    node$Value <- mean(data[,Y])
    bestSplit <- FindBestSplit( Y, Xnames, data, parentVal =  node[[type]], minobs = minobs, type = type )
    ifStop <- nrow( bestSplit ) == 0 
    if( node$depth == depth | ifStop | all( node$Prob %in% c(0,1) ) ){
      node$Leaf <- "*"
      return( node )
    }
    if(is.character(bestSplit$point)) {
      splitIndx <- data[, rownames(bestSplit)] %in% unlist(strsplit(bestSplit$point, ","))
    } else {
      splitIndx <- data[, rownames(bestSplit)] <= bestSplit$point
    }
    childFrame <- split( data, splitIndx )
    namel <- sprintf( "%s %s %s",  rownames(bestSplit), ifelse(is.na(as.numeric(bestSplit$point)), "is", "<="), bestSplit$point)
    childL <- node$AddChild( namel )
    childL$depth <- node$depth + 1
    childL[[type]] <- bestSplit$lVal
    BuildTree_bin( node = childL, Y = Y,  Xnames = Xnames, data = childFrame[["TRUE"]], type = type, depth = depth, minobs = minobs )
    name <- sprintf( "%s %s %s",  rownames(bestSplit), ifelse(is.na(as.numeric(bestSplit$point)), "is not", ">"), bestSplit$point )
    childR <- node$AddChild( name )
    childR$depth <- node$depth + 1
    childR[[type]] <- bestSplit$rVal
    BuildTree_bin( node = childR, Y = Y,  Xnames = Xnames, data = childFrame[["FALSE"]], type = type, depth = depth, minobs = minobs )
  }
}
Tree_bin <- function(Y, X, data, type, depth, minobs, overfit, cf) {
  if(!StopIfNot(Y, X, data, type, depth, minobs, overfit, cf)) {
    return()
  }
  tree <- Node$new( "Root" )
  AssignInitialMeasures(tree, Y, data, type, depth)
  BuildTree_bin(tree, Y = Y, Xnames = X, data = data, depth = depth, minobs = minobs, type = type, overfit = overfit, cf = cf)
  PruneTree <- function(){}
  AssignInfo(tree, Y, X, data, type, depth, minobs, overfit, cf)
  return(tree)
}

CrossValidation_tree_bin <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 15, nrow = 1))
  colnames(results) <- c("minobs","depth","type","dokladnosc_train","dokladnosc_test","dokladnosc_rpart","czulosc_train","czulosc_test","czulosc_rpart","precyzja_train","precyzja_test",'precyzja_rpart','specyficznosc_train','specyficznosc_test','specyficznosc_rpart')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla drzew ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla drzew z parametrami nr: ", i))
    print("")    
    minobs <- parTune[i,]$minobs
    depth <- parTune[i,]$depth
    type <- parTune[i,]$type
    df_results <- data.frame(matrix(ncol = 15, nrow = 1))
    colnames(df_results) <- c("minobs","depth","type","dokladnosc_train","dokladnosc_test","dokladnosc_rpart","czulosc_train","czulosc_test","czulosc_rpart","precyzja_train","precyzja_test",'precyzja_rpart','specyficznosc_train','specyficznosc_test','specyficznosc_rpart')
    for (i in 1:kFold){
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      X_train <- subset(data_train, select = -c(Y))
      y_train <- (subset(data_train, select = c(Y)))
      y_train_factor <- matrix(as.numeric(factor(unlist(y_train),levels=c(0,1)))-1)
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test_factor <- matrix(as.numeric(factor(unlist(y_test),levels=c(0,1)))-1)
      
      #model train
      tree_pred_train <- Tree_bin( Y = "Y", X = colnames(X_train), data = data_train, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_train <- PredictTree(tree_pred_train, data_train)
      predictions_train <- ifelse(predictions_train[,1] >= 0.5, 1, 0)
      predictions_train_factor <- factor(predictions_train, levels=c(0,1))
      #miary dokladnosci dla train
      dokladnosc_train <- dokladnosc(y_train_factor, predictions_train_factor)
      czulosc_train <- czulosc(y_train_factor, predictions_train_factor)
      results_train <- confusionMatrix( factor(unlist(y_train),levels=c(0,1)), predictions_train_factor )$table
      precyzja_train <- sum(diag(results_train)) / sum(apply(results_train, 2, sum))
      specyficznosc_train <- specyficznosc(y_train_factor, predictions_train_factor)
      #model test
      tree_pred_test <- Tree_bin( Y = "Y", X = colnames(X_test), data = data_test, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_test <- PredictTree(tree_pred_test, data_test)
      predictions_test <- ifelse(predictions_test[,1] >= 0.5, 1, 0)
      predictions_test_factor <- factor(predictions_test, levels=c(0,1))
      #miary dokladnosci test
      dokladnosc_test <- dokladnosc(y_test_factor, predictions_test_factor)
      czulosc_test <- czulosc(y_test_factor, predictions_test_factor)
      results_test <- confusionMatrix( factor(unlist(y_test),levels=c(0,1)), predictions_test_factor)$table
      precyzja_test <- sum(diag(results_test)) / sum(apply(results_test, 2, sum))
      specyficznosc_test <- specyficznosc(y_test_factor, predictions_test_factor)
      
      #model z pakietu rpart
      cf=0.2
      tree_caret <- rpart(Y ~., data = data_train, control = rpart.control(minsplit = minobs, maxdepth = depth, cp = cf))
      predictions_caret <- round(predict(tree_caret, data_test))
      predictions_caret_factor <- factor(predictions_caret, levels=c(0,1))
      #miary dokladnosci modelu
      dokladnosc_caret <- dokladnosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      czulosc_caret <- czulosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      precyzja_caret <- precyzja(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      specyficznosc_caret <- specyficznosc(factor(unlist(y_train),levels=c(0,1)), predictions_caret_factor)
      
      df_results <- rbind(df_results, c(0,0,0, dokladnosc_train,dokladnosc_test,dokladnosc_caret,czulosc_train,czulosc_test,czulosc_caret,precyzja_train,precyzja_test,precyzja_caret,specyficznosc_train,specyficznosc_test,specyficznosc_caret))
    }
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['minobs']] <- minobs
    df_results[['depth']] = depth
    df_results[['type']] = type
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}

#Wieloklasowa#
CrossValidation_tree_wiel <- function(dane, kFold, seed, parTune){
  results <- data.frame(matrix(ncol = 15, nrow = 1))
  colnames(results) <- c("minobs","depth","type","dokladnosc_train","dokladnosc_test","dokladnosc_rpart","recall_train","recall_test","recall_rpart","precyzja_train","precyzja_test",'precyzja_rpart','auc_train','auc_test','auc_rpart')
  set.seed(seed)
  indxT <- list()
  for(i in 1:kFold)
  {
    indxT[[i]] <- sample( x = 1:nrow(dane), 
                          size = (1-1/kFold)*nrow(dane),
                          replace = F )
  }
  for (i in 1:nrow(parTune)){
    k <- parTune[i,]
    print(paste0("Obliczam wyniki dla drzew ", colnames(parTune[i,]), "=", parTune[i,]))
    print(paste0("Obliczamy wyniki dla drzew z parametrami nr: ", i))
    print("")    
    minobs <- parTune[i,]$minobs
    depth <- parTune[i,]$depth
    type <- parTune[i,]$type
    df_results <- data.frame(matrix(ncol = 15, nrow = 1))
    colnames(df_results) <- c("minobs","depth","type","dokladnosc_train","dokladnosc_test","dokladnosc_rpart","recall_train","recall_test","recall_rpart","precyzja_train","precyzja_test",'precyzja_rpart','auc_train','auc_test','auc_rpart')
    for (i in 1:kFold){
      
      data_train <- dane[indxT[[i]],]
      data_test <- dane[-indxT[[i]],]
      X_train <- subset(data_train, select = -c(Y))
      y_train <- subset(data_train, select = c(Y))
      y_train_factor <- factor(unlist(y_train),levels=c("L", "B", "R"))
      y_train_numeric <- ifelse(y_train == "L", 1, ifelse(y_train =="B",2,3))
      X_test <- subset(data_test, select = -c(Y))
      y_test <- subset(data_test, select = c(Y))
      y_test_factor <- factor(unlist(y_test),levels=c("L", "B", "R"))
      y_test_numeric <- ifelse(y_test == "L", 1, ifelse(y_test =="B",2,3))
      
      #model train
      tree_pred_train <- Tree_bin( Y = "Y", X = colnames(X_train), data = data_train, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_train <- PredictTree(tree_pred_train, data_train[-1])$V4
      predictions_train_factor <- factor(predictions_train, levels=c("L", "B", "R"))
      predictions_train <-  ifelse( predictions_train_factor == "L", 1, ifelse( predictions_train_factor == "B", 2, 3))
      results_train <- confusionMatrix( y_train_factor, predictions_train_factor )$table
      #miary dokladnosci train
      dokladnosc_train <- sum(diag(results_train)) / sum(results_train)
      recall_train <- sum(diag(results_train)) / sum(apply(results_train, 1, sum))
      precyzja_train <- sum(diag(results_train)) / sum(apply(results_train, 2, sum))
      auc_train <- auc(multiclass.roc(y_train_numeric, predictions_train, levels=c(1,2,3)))[1]
      #model test
      tree_pred_test <- Tree_bin( Y = "Y", X = colnames(X_test), data = data_test, depth = depth, minobs = minobs, type=type, overfit = "none", cf = 0.2)
      predictions_test <- PredictTree(tree_pred_test, data_test[-1])$V4
      predictions_test_factor <- factor(predictions_test, levels=c("L", "B", "R"))
      predictions_test <-  ifelse( predictions_test_factor == "L", 1, ifelse( predictions_test_factor == "B", 2, 3))
      results_test <- confusionMatrix( y_test_factor, predictions_test_factor )$table
      #miary dokladnosci test
      results_test <- confusionMatrix((y_test_factor), predictions_test_factor)$table
      dokladnosc_test <- sum(diag(results_test)) / sum(results_test)
      recall_test <- sum(diag(results_test)) / sum(apply(results_test, 1, sum))
      precyzja_test <- sum(diag(results_test)) / sum(apply(results_test, 2, sum))
      auc_test <- auc(multiclass.roc(y_test_numeric, predictions_test, levels=c(1,2,3)))[1]
      
      #model z pakietu rpart
      cf=0.2
      tree_caret <- rpart(Y ~., data = data_train, control = rpart.control(minsplit = minobs, maxdepth = depth, cp = cf))
      predictions_caret <- predict(tree_caret, data_test,type='class')
      predictions_caret_factor <- factor(predictions_caret, levels=c("L", "B", "R"))
      predictions_caret <-  ifelse( predictions_caret_factor == "L", 1, ifelse( predictions_caret_factor == "B", 2, 3))
      results_caret <- confusionMatrix( y_test_factor, predictions_caret_factor )$table
      #miary dokladnosci rpart
      dokladnosc_caret <- sum(diag(results_caret)) / sum(results_caret)
      recall_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 1, sum))
      precyzja_caret <- sum(diag(results_caret)) / sum(apply(results_caret, 2, sum))
      auc_caret <- auc(multiclass.roc(y_test_numeric, predictions_caret, levels=c(1,2,3)))[1]
      
      df_results <- rbind(df_results, c(0,0,0, dokladnosc_train,dokladnosc_test,dokladnosc_caret,recall_train,recall_test,recall_caret,precyzja_train,precyzja_test,precyzja_caret,auc_train,auc_test,auc_caret))
    }
    df_results <- df_results[-1,]
    df_results<-colMeans(df_results)
    df_results[['minobs']] <- minobs
    df_results[['depth']] = depth
    df_results[['type']] = type
    results <- rbind(results, df_results)
  }
  results <- results[-1,]
  return(results)
}