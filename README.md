# moderation-tester
# Moderation Tester

Welcome to the Moderation Tester GitHub project! Moderation Tester is designed to help you evaluate responses from various models and providers by experimenting with different moderation settings. Adjust these settings prior to making requests and keep track of your findings.

## Features

- **Customisable Moderation Settings:** Tailor the moderation levels before sending your query to suit your requirements.
- **Model and Provider Evaluation:** Test and compare how different models respond under varied moderation constraints.
- **Document Upload:** Enhance your requests by uploading additional documents for comprehensive testing.

## Getting Started

To get started with Moderation Tester, follow these steps:

1. **Install Dependencies:** From the project folder, install Python dependencies:
    ```bash
    pip install flask requests
    ```

2. **Set API Keys:** Configure provider keys via environment variables:
    ```bash
    set OPENAI_API_KEY=your_key
    set GEMINI_API_KEY=your_key
    ```

3. **Run the Application:** Start the local web app:
    ```bash
    python moderation-tester.py
    ```

## Usage

- **Configure Moderation:** Adjust the moderation settings via the provided interface before sending any requests. This will help in assessing how moderation affects responses.
- **Upload Documents:** You may upload accompanying documents to test how each model handles additional data. The upload feature can be accessed on the main testing page.
- **Test Different Models:** Choose from a variety of models and providers to compare how different systems handle moderated inputs.
- **Internal Use Only:** This tool is intended for internal evaluation purposes.

## Contributing

We welcome contributions to help improve Moderation Tester. Please ensure that any pull requests adhere to our coding standards. For major changes, please open an issue first to discuss your ideas.

## Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for more details.

## Questions or Feedback

If you have questions or would like to provide feedback, please reach out through the Issues section of the GitHub repository.

Happy Testing!
