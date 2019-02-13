
# Read only region start

solution <- function(a, theta, X0, sigma) {
  # Input parameters are : 
  #  input1 As Numeric eg:- 232323.4323  
  #  input2 As Numeric eg:- 232323.4323  
  #  input3 As Numeric eg:- 232323.4323  
  #  input4 As Numeric eg:- 232323.4323
  
  # Expected return type  : 
  #  output1 As vector of type Numeric
  # Read only region end
  
  t <- 2
  #1 answer
  answer1 <- X0 * exp(-a*t) + theta*(1-exp(-a*t))
  
  #2 answer
  sim = 10^5
  vec<- rep(X0,sim)
  breaks = 100
  for(i in seq(1,breaks)){
    x<-rnorm(sim)
    vec<- vec+a*(theta - vec)*(t/breaks) + sigma*sqrt(vec)*x*sqrt(t/breaks);
  }
  for(i in seq(1,length(vec))){
    vec[i] = max(vec[i]-100,0)
  }
  answer2<-mean(vec)
  
  
  #3 answer
  del<-10^-7
  sigma1<-sigma+del
  vec<- rep(X0,sim)
  vec1<-rep(X0,sim)
  for(i in seq(1,breaks)){
    x<-rnorm(sim)
    vec<- vec+a*(theta - vec)*(t/breaks) + sigma*sqrt(vec)*x*sqrt(t/breaks);
    vec1<- vec1+ a*(theta - vec1)*(t/breaks) + sigma1*sqrt(vec1)*x*sqrt(t/breaks);
  }
  vega <- 0
  for(i in seq(1,sim)){
    if(vec[i]>100&&vec1[i]>100){
      vega <- vega+((vec1[i]-vec[i])/del)
    }
  }
  answer3<-vega/sim
  
  
  #OUTPUT	
  output <- c(answer1,answer2,answer3);
  return (output);
  stop("UnsupportedOperation solution( input1, input2, input3, input4)")
}
