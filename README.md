# README - Processador de Perguntas Matemáticas

## Descrição

Este projeto utiliza a API da OpenAI para receber perguntas matemáticas, validá-las e retornar uma resposta gerada por IA. Ele é estruturado como um fluxo de estado utilizando a biblioteca `langchain`.

## Requisitos

Antes de executar o projeto, certifique-se de ter instalado os seguintes requisitos:

- Python 3.8 ou superior
- Um ambiente virtual (recomendado)
- Dependências listadas no arquivo `requirements.txt`
- Uma chave de API da OpenAI

## Instalação

1. Clone este repositório:

   ```sh
   git clone https://github.com/viniciusferrazcastro/tutor-matematico-langchain.git
   cd seu-repositorio
   ```

2. Crie um ambiente virtual e ative-o:

   ```sh
   python -m venv venv
   source venv/bin/activate  # No Windows, use: venv\Scripts\activate
   ```

3. Instale as dependências:

   ```sh
   pip install -r requirements.txt
   ```

4. Configure a chave de API da OpenAI:

   - Crie um arquivo `.env` na raiz do projeto e adicione:
     ```sh
     OPENAI_API_KEY=your_openai_api_key
     ```

## Execução do Projeto

Para executar o programa, utilize:

```sh
python main.py
```

## Exemplo de Uso

O programa recebe uma pergunta matemática e retorna a resposta gerada pela IA.

### Entrada:

```sh
Enter your question: Solve the equation 2x + 3 = 7
```

### Saída:

```sh
Question Valid question! Sending to the Virtual Professor...

Virtual Professors Response:
Question: Solve the equation 2x + 3 = 7
Response:
x = 2
```

## Estrutura do Projeto

```
/
├── main.py  # Arquivo principal do projeto
├── requirements.txt  # Lista de dependências
├── .env  # Arquivo para configuração da API Key
└── README.md  # Instruções de uso
```

## Licença

Este projeto é distribuído sob a Licença **MIT. Sinta**-se à vontade para modificar e usar conforme necessário.

