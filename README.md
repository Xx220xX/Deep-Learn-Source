Uma rede neural rapida e otimazada
DNN (deep neural network, em portugues rede neural profunda )

## Como usar?
Primeiro instancie uma DNN .
```c
    int arquitetura[] = {2,5,5,1};
    int func_id[] = {TANH,SOFTMAX,RELU};
    DNN dnn = newDnnWithFunctions(arquitetura,4,func_id);
```
Preencha os pesos com valores aleatorios
```c
    setSeed(time(NULL));
    randomize(dnn);
```
Prepare os dados para treinamento,
esta biblioteca não faz isso, você deve implementar ao seu modo.

```c
    // LEIA OS EXEMPLOS E OS PREPARE PARA A REDE
    double **entrada;
    double **saida;
    int numero_exemplos;
    ...//nesta parte vc le os dados de entrada e saida da 
```
Agora treine a rede por algumas epocas
```c
    int max_epocas = 10000;
    for(int epoca=0;epoca<max_epocas;++epoca){
        for(int exemplo=0;exemplo<numero_exemplos;++exemplo){
            // propagation
            call(dnn,entrada[exemplo]);
            // back-propagation
            learn(dnn,saida[exemplo]);
        }
    }   
```

Após treinar você pode avaliar o quão boa esta a sua rede
```c
    // cria vetor com o mesmo tamanho da saida da rede
    double *saida_rede = (double *)calloc(sizeof(double),arqutetura[3]);
    double **entrada_teste,**saida_teste;
    int numero_teste;
    ...// prepare os dados do teste
    int acertos = 0; 
    for(int teste=0;teste<numero_teste;++teste){
        call(dnn,entrada_teste[teste]);
        getOut(dnn,saida_rede);
        // utilize saida_rede e saida_teste[teste] para verificar se a rede acertou
        ...
    }
```
É possivel salvar os pesos da rede treinada
```c
    saveDNN(dnn,"rede_treinada.dnn");
```
Depois libere os recursos utilizados no programa
```c
    // libera recursos usado pela rede neural
    releaseDNN(dnn);
    // libere os recursos usados para instanciar seus dados
    ...
```
## Documentação 
Fica disponivel o arquivo de cabeçalho gabriela.h
contendo as funções:

##### newDnn
<a name=newDNN></a>
Cria uma rede neural profunda com os parametros passado.
###### argumentos:
* int *arquitetura, int tamanho_arquitetura, double taxa_aprendizado

    - arquitetura: é um vetor não nulo de tamanho maior ou igual a 2 do tipo inteiro.
    Indica a quantidade de neuronios por camada e a quantidade de camadas que a rede neural irá conter
    - tamanho_arquitetura: indica o tamanho do vetor  arquitetura.
    - taxa_aprendizado: a taxa de aprendizado da rede neural.
###### retorno:
    Retorna um ponteiro do tipo DNN pronto para uso.
###### exemplo:
```c
    int arquitetura[]={2,5,5,1};
    int tamanho = sizeof(arquitetura)/sizeof(int);
    DNN dnn = newDNN(arquitetura,tamanho,0.1); 
```
    Exemplo para uma rede com 2 neuronios na camada de entrada e 1 neuronio na camada de saída
##### newDnnWithFunctions
   veja também [newDNN](newDNN), semelhante a newDNN porém neste é exigido o vetor
   de indentificador de função de cada camada.
###### argumentos:
* int *arquitetura, int tamanho_arquitetura,int *funcao_de_cada_camada, double taxa_aprendizado

    - arquitetura: é um vetor não nulo de tamanho maior ou igual a 2 do tipo inteiro.
    Indica a quantidade de neuronios por camada e a quantidade de camadas que a rede neural irá conter
    - tamanho_arquitetura: indica o tamanho do vetor  arquitetura.
    - funcao_de_cada_camada: deve possuir tamanho de  tamanho_arquitetura-1, contendo os identificadores 
    de função de cada camada. Veja a [tabela de funções disponiveis](funcoes)
    - taxa_aprendizado: a taxa de aprendizado da rede neural.
###### retorno:
    Retorna um ponteiro do tipo DNN pronto para uso.
###### exemplo:
```c
    int arquitetura[]={2,5,5,1};
    int func_id[]={TANH,SIGMOID,RELU};
    int tamanho = sizeof(arquitetura)/sizeof(int);
    DNN dnn = newDnnWithFunctions(arquitetura,tamanho,func_id,0.1); 
```   

##### releaseDNN
Libera os recursos usados pela rede neural, após usar essa função 
a instancia DNN passada nao existirá mais.
###### argumentos:
   * DNN dnn
     -  dnn: instancia para ser destruida 
###### retorno:
    Caso a função conclua com sucesso o valor retornado é 0.
    Caso o dnn seja NULL o valor retornado é -1
###### exemplo:
```c
    releaseDNN(dnn); 
```   
##### setSeed
Muda a semente usada para gerar os numeros pseudo aleatorios.
###### argumentos:
   * long long int semente
      - semente: semente a ser utilizada 
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
###### exemplo:
```c
    setSeed(time(NULL));//recomendado
```   
##### randomize
Aplica valores aleatorios nas matrizes de pesos da rede neural
###### argumentos:
   * DNN dnn
     - dnn: rede neural  
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
    caso dnn seja NULL o valor retornado é -1
###### exemplo:
```c
    randomize(dnn);
```   
##### call
<a name=call></a>
Realiza o forward-propagation na rede.
Veja [learn](learn) e [getOut](getout). 
###### argumentos:
   * DNN dnn, double *entrada
        - dnn: rede neural 
        - entrada: vetor double não nulo com o mesmo tamanho da primeira posição da arquitetura contendo os valores da entrada da rede 
    
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
    caso dnn seja NULL o valor retornado é -1
###### exemplo:
```c
    ...
    call(dnn,entrada);
```   
##### learn
<a name=learn></a>
Realiza o back-propagation na rede, é importante realizar o propagation primeiro
veja [call](call).
###### argumentos:
   * DNN dnn, double *saida_esperada
        - dnn: rede neural 
        - saida_esperada: vetor double não nulo com o mesmo tamanho da ultima posição da arquitetura contendo os valores que a rede deve obter na saída. 
    
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
    caso dnn seja NULL o valor retornado é -1
###### exemplo:
```c
    ...
    learn(dnn,saida_esperada);
```   

##### getOut
<a name=getout></a>
Pega os valores de saída da rede.
###### argumentos:
   * DNN dnn, double *buff_saida
        - dnn: rede neural 
        - buff_saida: vetor double não nulo com o mesmo tamanho da ultima posição da arquitetura, a função irá copiar os valores de saída para este vetor. 
    
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
    caso dnn seja NULL o valor retornado é -1
###### exemplo:
```c
    ...
    getOut(dnn,saida);
```   
##### saveDNN
Salva os pesos da rede neural em um arquivo,
O arquivo será salvo em modo binario, com um cabeçalho explicando a formatação de escrita dos dados
###### argumentos:
   * DNN dnn, char *nome_arquivo_saida
        - dnn: rede neural 
        - nome_arquivo_saida: string com o nome do arquivo de saída. 
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0
    caso aconteça algum erro a saida é diferente de 0.
###### exemplo:
```c
    ...
    saveDNN(dnn,"rede_aprendida.dnn");
```   

##### loadDNN
Lê os pesos salvos de uma rede neural em um arquivo.
###### argumentos:
   * DNN dnn, char *nome_arquivo_leitura
        - dnn: rede neural 
        - nome_arquivo_leitura: string com o nome do arquivo para leitura. 
###### retorno:
    caso a função conclua com sucesso o valor retornado é uma rede neural do tipo DNN pronta para uso
    caso aconteça algum erro o valor retornado é NULL
###### exemplo:
```c
    DNN dnn = loadDNN("rede_aprendida.dnn");
```   

##### setFuncID
Muda a função de ativação de uma camada.
###### argumentos:
   * DNN dnn, int camada, int ID_da_funcao
        - dnn: rede neural 
        - camada: indica qual camada que terá sua funcão modificada. deve ser maior ou igual a 2 e menor que o numero de camadas
        a primeira camada não possui função de ativação
        - ID_da_funcao: identificador da função de ativação a ser usada, consulte a [tabela](funcoes) 
###### retorno:
    caso a função conclua com sucesso o valor retornado é 0 
    caso aconteça algum erro o valor retornado é diferente de 0
###### exemplo:
```c
    // rede com arquitetura (2,5,5,1)
    setFunId(dnn,4,SOFTMAX);
```   

##### printDNN  
Mostra as configurações da rede neural na tela

###### argumentos:
   * DNN dnn
        - dnn: rede neural, caso seja NULL nada é mostrado 
         
###### retorno:
    não possui retorno
###### exemplo:
```c
    ...
    printDNN(dnn);
```   



### [Tabela de funções de ativação](funcoes)
| definição da função | MACRO | ID |
| :---- | :---- | :----: |
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\begin{cases}\ln(x+1.14)&\quad%20x\in(0.5,\infty)\\\-ln(-x+1.14)&\quad%20x\in(-\infty,-0.5)\\\tanh(x)%20&\quad%20x\in[-0.5,0.5]\\\end{cases}"/>| EXPERIMENTAL | 0 |
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=x"/> | IDENTIDADE | 1
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\frac{1}{1-e^{-x}}"/> | SIGMOID | 2
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\tanh(x)"/> | TANH | 3
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\begin{cases}x&\quad%20x\geq0\\0%20&\quad\text{otherwise}\\\end{cases}"/>| RELU | 4 |
| <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\frac{e^x}{\sum%20e^{a_i}}"/> | SOFTMAX | 5 |
