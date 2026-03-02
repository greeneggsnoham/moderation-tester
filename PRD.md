# Product Requirements Document (PRD) for Moderation Tester

## Purpose and Objective

The primary aim of the Moderation Tester is to evaluate how much of an improvement the changes to moderation will deliver to our core users. It serves as a tool for assessing the efficacy of modifications in moderation settings across different models and providers.

## Target Audience

- **Primary Users:** Product Management Team and Edina Management Team
- **Secondary Users:** IT Ethics Board

## Core Features and Requirements

1. **Moderation Settings**
   - Users can toggle moderation on or off.
   - For models that allow different levels or categories of moderation, users can set these levels and categories accordingly.

2. **Document Upload**
   - Users can upload documents in PDF, CSV, or Word formats to include with their prompts.

3. **User Interface**
   - A spinner activates while waiting for the response from the LLM.
   - Responses are displayed in a chat-like interface, including any moderation responses received.

4. **Provider Selection**
   - Not relevant at this time

## Technology and Platform

- **Programming Language:** Python
- **Execution Environment:** Runs locally on a PC
- **Interface:** Accessible via a web browser
- **Integration:** Utilises API keys for communication with providers

## Documentation and Support

- Minimal user documentation will be provided to guide initial setup and usage.
- The product management team is responsible for any technical support if needed.

## Compliance and Legal Considerations

- Require review by the IT Ethics Board to ensure alignment with ethical and legal standards, particularly concerning data handling and moderation policies.

## Testing and Quality Assurance

- Internal testing will be conducted by the product management team before the tool's deployment for management's evaluation.

## Privacy and Security

- All user interactions and document uploads will be handled securely, with considerations for API key management.
- Ensure robust security practices for any data storage or processing.

## Constraints

- This tool is designed for internal evaluation purposes and not intended for public deployment.
- UX is a secondary priority, focusing primarily on functionality.

This PRD sets the foundation for developing the Moderation Tester, guiding the team to deliver a tool that meets our objectives and aligns with user expectations. If further details or adjustments are required, please advise.
