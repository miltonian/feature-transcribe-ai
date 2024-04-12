
![feature-transcribe-ai-logo](https://github.com/miltonian/feature-transcribe-ai/assets/8435923/9eb97133-4949-4b62-b110-932928c9b6db)

# FeatureTranscribeAI

FeatureTranscribeAI is an innovative tool designed to streamline the development process by automating the identification and generation of code based on new feature requests. Utilizing the power of machine learning and the OpenAI API, FeatureTranscribeAI does the following: 
- Generates embeddings from your codebase
- Analyzes user input for feature requests
- Identifies relevant existing code
- Crafts prompts for generative models to automatically generate code fulfilling the feature request.

## Features

- **Automated Embeddings Generation**: Generate embeddings from your existing codebase to understand and analyze your code at a deeper level.
- **Feature Request Processing**: Accepts user inputs for new features and uses machine learning to understand the request.
- **Relevant Code Identification**: Finds relevant code in your codebase that aligns with the feature request, using generated embeddings and machine learning algorithms.
- **Code Generation for New Features**: Crafts prompts based on identified relevant code and the feature request for generative models, facilitating the automatic generation of code snippets to implement new features.

## Components

- **feature_to_code.py**: Analyzes embeddings to identify code snippets relevant to a new feature request. It handles loading embeddings, code content, and new feature descriptions.
- **openai_api.py**: Interfaces with the OpenAI API for generating embeddings from text inputs, crucial for analyzing both existing code and feature requests.
- **prepare_code.py**: Prepares your codebase for analysis by parsing and generating embeddings, integrating functionalities from `openai_api` and other parsing utilities.
- **code_parser.py**: Specializes in parsing Swift/Typescript/Javascript/Python files, enabling detailed analysis and embeddings generation for Swift/Typescript/Javascript codebases.
- **utils.py**: Offers utility functions for file and directory management, supporting the overall workflow of FeatureTranscribeAI.

## Requirements

- Python 3.x
- OpenAI API Key
- Required Python libraries (`numpy`, `requests`, etc.)

## Setup

1. Install Python 3.x and ensure it's correctly set up on your system.
2. Install necessary Python packages: `pip3 install -r requirements.txt`.
3. Configure your OpenAI API key as an environment variable for secure access: `export OPENAI_API_KEY='your_api_key_here'`.
4. To run the html example: `flask --app app.py run --port 5003`

## Usage

Using FeatureTranscribeAI involves just a couple of straightforward steps. Before starting, ensure your environment is set up as described in the **Setup** section.

1. **Prepare Your Codebase**: Run the `prepare_code.py` script to analyze your codebase and generate embeddings. This process respects your `.gitignore` settings, automatically excluding files and directories you've opted not to track with Git. This ensures a focus on the meaningful parts of your codebase for analysis.

   ```bash
   python3 feature_transcribe/prepare_code.py --directory '/path/to/your/code' 
This step parses your code (including Swift, TypeScript, JavaScript, and their JSX/TSX variants), generating embeddings that capture the semantic essence of your codebase.

2. **Identify Relevant Code**: Once your codebase is prepared, run `feature_to_code.py` with a description of the new feature you want to integrate. The script uses the generated embeddings to find code snippets that are most relevant to your feature request.

    ```bash
    python3 feature_transcribe/feature_to_code.py --feature 'feature_description.json'
 This command analyzes the embeddings to match your feature request with existing code, helping you identify where and how to best integrate new features.

### Respecting .gitignore

FeatureTranscribeAI automatically skips over files and directories specified in your project's `.gitignore` file during the embeddings generation and relevant code identification process. This ensures that temporary files, dependencies, or any other non-relevant code specified in `.gitignore` are not included in the analysis, keeping the focus on the meaningful parts of your codebase.

This feature helps maintain the integrity and relevance of the embeddings and code suggestions, providing more accurate and useful results based on the content you actively maintain in your repository.

## Contributing

Contributions to FeatureTranscribeAI are highly encouraged. Whether it's by enhancing functionality, adding new features, or fixing bugs, your input is valuable. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

FeatureTranscribeAI is released under the MIT License. For more details, see the LICENSE file in the repository.
