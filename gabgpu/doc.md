# DOCUMENTAÇÃO

### Dependencias OpenCl API
    Usado para computação paralela. 
    Outra API semelhante CUDA, porem é limitado a placas de vidio NVIDEA
  
### [GABRIELA](gabriela.h)
    Este cabeçalho contem as funções necessarias para o funcionamento da rede neural.
    Dentre elas :

 ```c
       int create_DNN(Gab *p_gab, int *arq, int l_arq, double hitLean);
       
       void initGPU(const char *src);
       
       void initWithFile(const char *filename);
       
       void endGPU();
       
       void setworkSizes(int maxWorks);
       
       void call(Gab *p_gab, double *inp);
       
       void learn(Gab *p_gab, double *trueOut);
       
       void release(Gab *p_gab);
       
       void getoutput(Gab *p_gab, double *out);
       
       void setSeed(unsigned long int seed);
       
       int randomize(Gab *p_gab);
 ```
 ### EXEMPLO 
```c
     
     initWithFile(file);// arquivo kernel.cl
     Gab gab; // struct para armazenar dados da rede neural
     int arquitetura[] = {2, 18,9,3, 1}; // arquitetura da rede
     int la = sizeof(arquitetura) / sizeof(int); // tamanho da arquitetura
     create_DNN(&gab, arquitetura, la, 0.1); // cria a rede neural
     double input[][2] = {{1, 1},{0, 1},{1, 0},{0, 0}};// cria entrada para exemplo xor
     double out[4][1];  
     for (int i = 0; i < 4; ++i) out[i][0] = op(input[i][0], input[i][1]);//cria objetivo para exemplo xor
     
     int epocas = 0; 
     int maxEpocas = 100; // maximo de epocas
     int i; 
     double o = 0.0; // saida da rede
     double minEnergia = 1e-3; // energia minima para criterio de parada(geralmente nao é usado)
     double energia = minEnergia + 1;
     clock_t t0 = clock();
     for (epocas = 0; energia > minEnergia && epocas < maxEpocas; epocas++) {
         for (i = 0; i < 4; ++i) {
             call(&gab, input[i]);// propagation
             learn(&gab, out[i]); // backpropagation
             getoutput(&gab, &o);// pega saida
             energia += (o - out[i][0]) * (o - out[i][0]); // faz o calculo do erro 
         }
         energia /= 2.0;
     }
     t0 = clock() - t0;
     printf("depois de %d epocas a energia foi %lf\n", epocas, energia);
     printf("tempo total gasto %ld ms\n", t0);
     printf("tempo medio gasto por exemplo %lf ms\n", t0 / ((double) epocas * 4.0));
     endGPU();// libera recursos da gpu
```