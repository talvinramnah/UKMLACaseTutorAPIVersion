# UKMLA Case Tutor API - Structured OpenAI Assistant Output Upgrade

## Background and Motivation
The current system uses OpenAI's Assistant API to generate medical cases and feedback for users, but the responses are unstructured free text. This leads to inconsistencies, makes it harder to parse or display information in the frontend, and limits the ability to provide nuanced, actionable feedback or handle refusals gracefully. By adopting OpenAI's structured outputs (JSON mode), we can ensure:

- Consistent, machine-readable case structure.
- Clear separation of case components (demographics, history, ICE, etc.).
- More actionable, granular feedback for users.
- Smooth, elegant refusals to nonsense or inappropriate responses.
- Easier admin control and simulation of full cases.

## Key Challenges and Analysis
- **Prompt Engineering:** The Assistant must be prompted to return JSON objects for each stage (case intro, question, feedback, etc.), and to handle refusals in a structured way.
- **API Integration:** The FastAPI backend must be updated to parse and stream JSON responses, not just free text.
- **Backward Compatibility:** Ensure the new structure does not break existing frontend expectations, or provide a migration plan.
- **Admin Simulation:** Provide a way for admins to simulate a full case run-through for testing and demonstration.
- **Feedback Structure:** Feedback must be broken down into "What went well," "What can be improved," and "Actionable points," each with subcategories for management and investigation.
- **Pass/Fail Logic:** The Assistant must use its own judgment to assign pass/fail at the end, based on user performance.
- **Example Case Referencing:** Ensure the Assistant can reference uploaded example cases as context.

## High-level Task Breakdown

### 1. **Design JSON Schemas for Case and Feedback**
   - Define the JSON structure for:
     - Initial case message (demographics, history, ICE).
     - Each question/response cycle.
     - End-of-case feedback (with pass/fail and breakdown).
     - Refusal/validation errors.
   - **Success Criteria:** Schemas are documented and reviewed.

### 2. **Update Assistant Prompting for Structured Output**
   - Rewrite the system prompt for the Assistant to:
     - Always respond in the defined JSON format.
     - Gently guide users, giving hints after 2 failed attempts, and answers after 3.
     - Use pass/fail at the end, with structured feedback.
     - Handle admin simulation command.
     - Reference example cases if needed.
   - **Success Criteria:** Prompt is clear, robust, and covers all requirements.

### 3. **Update Backend to Parse and Stream JSON Responses**
   - Update `/start_case` and `/continue_case` endpoints to:
     - Parse and validate JSON responses from the Assistant.
     - Stream JSON chunks to the frontend.
     - Handle and display structured refusals.
   - **Success Criteria:** Backend can handle and stream structured JSON responses without errors.

### 4. **Implement Admin Simulation Command**
   - Add a special command (e.g., `/simulate_full_case`) that triggers the Assistant to run through a full case automatically.
   - Ensure this is only accessible to admins.
   - **Success Criteria:** Admin can simulate a full case and receive the full JSON output.

### 5. **Update Feedback and Pass/Fail Logic**
   - Ensure the Assistant's end-of-case feedback is broken down as specified.
   - Update backend and frontend to display this feedback clearly.
   - **Success Criteria:** Feedback is always structured and actionable, and pass/fail is clear.

### 6. **Test and Validate with Example Cases**
   - The example files are already uploaded in the vector store under the following filenames:
     - EXAMPLE 1-Acute Heart Failure Management .txt
     - EXAMPLE 2- Mitral stenosis.txt
     - EXAMPLE 3- Cardiac tamponade .txt
   - Next: Test that the Assistant can reference these in its responses when relevant.
   - Validate that responses are contextually appropriate and reference examples as needed.
   - Confirm that all previous tasks (JSON structure, feedback, admin simulation) work with example-based context.

### 7. **Frontend/UX Review (Optional)**
   - (If required) Update frontend to display new structured responses elegantly.
   - **Success Criteria:** Users see clear, structured case info and feedback.

## Project Status Board

- [ ] 1. Design JSON Schemas for Case and Feedback
- [ ] 2. Update Assistant Prompting for Structured Output
- [ ] 3. Update Backend to Parse and Stream JSON Responses
- [ ] 4. Implement Admin Simulation Command
- [ ] 5. Update Feedback and Pass/Fail Logic
- [ ] 6. Test and Validate with Example Cases
- [ ] 7. (Optional) Frontend/UX Review

---

## Task 1: Design JSON Schemas for Case and Feedback

### 1. Initial Case Message Schema
```jsonc
{
  "demographics": {
    "name": "string",           // Patient's first name (randomized)
    "age": "integer",           // Patient's age
    "nhs_number": "string",     // Random NHS number (format: 10 digits)
    "date_of_birth": "string",  // ISO date string (YYYY-MM-DD)
    "ethnicity": "string"       // Ethnicity (e.g., "White British", "South Asian")
  },
  "presenting_complaint": {
    "summary": "string",        // One-line presenting complaint (SOCRATES)
    "history": "string",        // History of presenting complaint
    "medical_history": "string",// Relevant past medical history
    "drug_history": "string",   // Relevant drug history
    "family_history": "string"  // Relevant family history
  },
  "ice": {
    "ideas": "string",          // Patient's ideas
    "concerns": "string",       // Patient's concerns
    "expectations": "string"    // Patient's expectations
  }
}
```

### 2. Question/Response Cycle Schema
```jsonc
{
  "question": "string",           // The question posed to the user
  "attempt": "integer",           // Attempt number (1, 2, or 3)
  "user_response": "string",      // User's answer (if available)
  "assistant_feedback": "string", // Assistant's feedback or hint
  "is_final_attempt": "boolean",  // True if this is the last attempt before answer is revealed
  "correct_answer": "string",     // Provided only after 3rd failed attempt
  "next_step": "string"           // Description of what happens next in the case
}
```

### 3. End-of-Case Feedback Schema
```jsonc
{
  "result": "pass" | "fail",      // Pass or fail
  "feedback": {
    "what_went_well": {
      "management": "string",      // What went well in management
      "investigation": "string",   // What went well in investigation
      "other": "string"            // Any other positive points
    },
    "what_can_be_improved": {
      "management": "string",      // Improvements in management
      "investigation": "string",   // Improvements in investigation
      "other": "string"            // Other areas for improvement
    },
    "actionable_points": [
      "string"                      // List of actionable suggestions (e.g., "Do more cases in Cardiology ward")
    ]
  }
}
```

### 4. Refusal/Validation Error Schema
```jsonc
{
  "error": {
    "type": "refusal" | "validation_error", // Type of error
    "message": "string"                      // Human-readable error message
  }
}
```

// End of Task 1 draft schemas

---

## Task 2: Update Assistant Prompting for Structured Output

### Draft System Prompt for OpenAI Assistant (JSON Mode)

```text
You are an expert UK medical educator and case simulator. You must always respond in valid JSON according to the following schemas:
- Initial case message (demographics, presenting complaint/history, ICE)
- Question/response cycle
- End-of-case feedback (with pass/fail and breakdown)
- Refusal/validation error

## Instructions:
1. When starting a new case, respond with a JSON object containing:
   - demographics: name (random), age, NHS number (random 10 digits), date of birth (ISO), ethnicity
   - presenting_complaint: summary (SOCRATES), history, medical_history, drug_history, family_history
   - ice: ideas, concerns, expectations
   (See schema for field names and types)

2. Guide the user through the case step-by-step. For each question:
   - Pose a question in the JSON format
   - Wait for the user's response
   - If the user answers incorrectly, allow up to 2 attempts, providing a gentle hint after the 2nd failed attempt
   - On the 3rd failed attempt, provide the correct answer and move to the next question
   - If the user answers correctly, acknowledge and proceed
   - Always use the question/response JSON schema

3. At the end of the case, provide a JSON object with:
   - result: "pass" or "fail" (see below for criteria)
   - feedback: what_went_well (management, investigation, other), what_can_be_improved (management, investigation, other), actionable_points (list)
   - Pass if the user managed and investigated the patient safely and effectively. Fail if the user would have killed the patient, caused significant harm, or required someone else to take over. Use your best judgment for borderline cases.

4. If the user sends the admin command `/simulate_full_case`, simulate the entire case from start to finish, including all questions, answers, and feedback, in a single JSON response (or as a sequence of valid JSON objects if needed for streaming).

5. If the user input is nonsense, inappropriate, or cannot be answered, respond with a refusal/validation error JSON object, explaining the issue.

6. You may reference uploaded example cases (e.g., "EXAMPLE 1-Acute Heart Failure Management.txt") if relevant to the current case or to provide context.

7. Never output free text or markdown. Only output valid JSON objects as per the schemas.

## Example (Initial Case Message):
{
  "demographics": { ... },
  "presenting_complaint": { ... },
  "ice": { ... }
}

## Example (Question/Response):
{
  "question": "What is the first investigation you would order?",
  "attempt": 1,
  "user_response": "",
  "assistant_feedback": "",
  "is_final_attempt": false,
  "correct_answer": "",
  "next_step": ""
}

## Example (End-of-Case Feedback):
{
  "result": "pass",
  "feedback": { ... }
}

## Example (Refusal):
{
  "error": {
    "type": "refusal",
    "message": "Your input was not understood. Please try again."
  }
}
```

// End of Task 2: System prompt draft

---

## Task 3: Update Backend to Parse and Stream JSON Responses

### Subtasks
- [x] Outline backend changes and test plan (this section)
- [x] Update `/start_case` endpoint to expect and validate JSON responses from the Assistant (using Task 1 schemas)
- [x] Update `/continue_case` endpoint to expect and validate JSON responses from the Assistant
- [x] Update streaming logic to handle JSON chunks and errors (including refusals/validation errors)
- [x] Ensure structured refusals/validation errors are handled and surfaced to the frontend
- [ ] Add/expand backend tests for JSON handling (in progress)

**Current:**
- Planning and implementing backend unit/integration tests for JSON validation and streaming

### Backend Test Outline
- Unit tests for `validate_initial_case_json`, `validate_question_response_json`, and `validate_feedback_json` (valid and invalid cases)
- Mock streaming generator to simulate Assistant JSON output and test chunk parsing/validation
- Integration test: Simulate full `/start_case` and `/continue_case` flows with valid and invalid JSON
- Error test: Simulate refusal/validation error JSON and ensure correct frontend display

// End of Task 3 outline

---

## Task 4: Implement Admin Simulation Command

- The admin simulation command will be triggered if any user types 'SpeedRunGT86' as their input.
- When triggered, the Assistant should simulate the entire case in one go, streaming the full JSON output (initial case, all questions/answers, and feedback).
- No special admin authentication is required; any user can trigger this for now.
- This will be handled in the `/continue_case` endpoint logic.

**Status:** Complete

---

## Task 5: Update Feedback and Pass/Fail Logic

- The Assistant should always use 'pass' or 'fail' (never a score) in the end-of-case feedback JSON.
- Feedback must be structured as per the schema: what_went_well, what_can_be_improved, actionable_points (with management/investigation/other breakdowns).
- Ensure backend and frontend display this feedback clearly.
- Test with both normal and admin-simulated cases.

**Status:** Complete

---

## Task 6: Test and Validate with Example Cases

- The example files are already uploaded in the vector store under the following filenames:
  - EXAMPLE 1-Acute Heart Failure Management .txt
  - EXAMPLE 2- Mitral stenosis.txt
  - EXAMPLE 3- Cardiac tamponade .txt
- Next: Test that the Assistant can reference these in its responses when relevant.
- Validate that responses are contextually appropriate and reference examples as needed.
- Confirm that all previous tasks (JSON structure, feedback, admin simulation) work with example-based context.

**Status:** In progress

---

# UKMLA Case Tutor API - Add Desired Specialty Field

## Background and Motivation
The user has added a `desired_specialty` column to the `user_metadata` table in Supabase and updated the frontend onboarding modal to capture this information. The backend onboarding endpoints need to be updated to accept and store this new field.

## Key Challenges and Analysis
- Need to update the `OnboardingRequest` Pydantic model to include the `desired_specialty` field
- Need to update the `/onboarding` endpoint to handle and store the new field
- Need to update the `/user_metadata/me` endpoint to return the new field
- Ensure proper validation for the specialty field

## High-level Task Breakdown
- [x] Update `OnboardingRequest` model to include `desired_specialty` field
- [x] Update `/onboarding` endpoint to store `desired_specialty` in database
- [x] Update `/user_metadata/me` endpoint to return `desired_specialty` field
- [ ] Test the changes to ensure they work correctly

## Project Status Board
- [x] **Task 1**: Update OnboardingRequest Pydantic model ✅
- [x] **Task 2**: Update /onboarding endpoint database insertion ✅
- [x] **Task 3**: Update /user_metadata/me endpoint response ✅
- [ ] **Task 4**: Test the implementation
- [x] Fix CORS allow_origins list to include both https://bleep64.com and https://www.bleep64.com (missing comma bug)
- [ ] Redeploy/restart FastAPI server to apply CORS changes
- [x] Fix /continue_case streaming so 'is_completed' is only sent when the case is truly finished

## Current Status / Progress Tracking
- Status: Streaming fix for 'is_completed' in /continue_case implemented
- Completed: Now, 'is_completed' is only included in the response when the case is truly finished (i.e., after '[CASE COMPLETED]' is detected in the final message). For all intermediate steps, it is omitted.
- Next: Test the streaming endpoint with a real case to ensure the frontend only shows feedback and post-case actions at the correct time.

## Executor's Feedback or Assistance Requests
**Implementation Complete**: I have successfully updated the backend to handle the `desired_specialty` field:

1. **OnboardingRequest Model**: Added `desired_specialty: str = Field(..., min_length=1, max_length=100)` to the Pydantic model
2. **Onboarding Endpoint**: Updated the database insertion to include `"desired_specialty": request.desired_specialty` in the `insert_data` dictionary
3. **User Metadata Endpoint**: Updated the select query to include `desired_specialty` in the returned fields

The changes are minimal and focused, maintaining consistency with the existing code patterns. The field validation ensures it's required (not optional) and has appropriate length constraints.

**Ready for Testing**: The implementation should now work seamlessly with the frontend code that's already sending the `desired_specialty` field.

- The missing comma between 'https://bleep64.com' and 'https://www.bleep64.com' in the CORS allow_origins list has been fixed. Please redeploy or restart the FastAPI server for the change to take effect. After redeployment, test login and API requests from both https://bleep64.com and https://www.bleep64.com to confirm CORS is working.

## Lessons
- The backend was well-structured, making it easy to add the new field by following existing patterns
- All three components (model, insertion, retrieval) needed to be updated for complete functionality
- The Supabase REST API approach used in the onboarding endpoints made the changes straightforward
- Always check for missing or extra commas in Python lists, especially in configuration sections like CORS allow_origins. A missing comma can silently break CORS and is easy to overlook.
- Only include 'is_completed': true in the streaming response when the case is truly finished. Including it too early causes the frontend to end the chat and show feedback prematurely.

---

## Planner: Optimizing Structured JSON Case Streaming (UKMLA Case Tutor)

## Background and Motivation
- The goal is to deliver a structured, stepwise medical case to the frontend using JSON objects for each logical step (demographics, history, questions, feedback, etc.).
- The backend and frontend are set up to stream and display each JSON object as a separate SSE event.
- Recent changes include a more concise system prompt to the OpenAI Assistant, aiming to reduce the time to first response.

## Key Issues Observed
- **Initial case load is slow (16+ seconds):** The Assistant generates the entire intro as a single JSON object, causing a delay before anything is shown.
- **No true streaming for the intro:** All content appears at once, not incrementally.
- **Frontend UX issues:**
  - Loading bubble remains after case loads.
  - **Response textbox does not render when the initial assistant response (case intro) has been returned.**
- **Prompt size/complexity:** Large prompts may increase latency; a more concise prompt may help.

## Goals
- Reduce the time to first meaningful content for the user (ideally <10s).
- Preserve structured JSON output for robust frontend parsing and display.
- Improve perceived streaming and frontend responsiveness.
- Ensure the frontend removes the loading bubble and shows the response textbox at the right time.

## Analysis
- The OpenAI Assistant, when prompted to output a single JSON object for the intro, will not stream partial content; the backend can only yield when the full object is received.
- A more concise prompt may reduce model latency, but the initial case will still be a single JSON object.
- True incremental streaming of the intro would require the Assistant to output each section (demographics, history, ICE, etc.) as separate JSON objects, which is not the current prompt design.
- Frontend logic must be robust to remove loading indicators and show the input box as soon as the first question is received. **Currently, the response textbox does not render after the initial assistant response, which blocks user interaction.**

## High-Level Task Breakdown

1. **Monitor and Benchmark Initial Response Time**
   - Measure time from request to first SSE event for various prompt sizes.
   - Success: Initial case intro appears in <10s for most cases.

2. **Optimize System Prompt for Latency**
   - Keep instructions concise but explicit about required JSON structure.
   - Remove unnecessary examples or verbose text.
   - Success: No loss of structure, but reduced latency.

3. **(Optional/Future) Explore Incremental Streaming**
   - Experiment with prompting the Assistant to output each section as a separate JSON object (demographics, then history, then ICE, etc.).
   - Backend yields each as soon as received.
   - Success: Perceived streaming for the intro.

4. **Frontend UX Improvements**
   - Remove the loading bubble as soon as the first real assistant message is received.
   - **Ensure the response textbox renders as soon as the assistant is ready for user input (e.g., after the first question is received).**
   - Success: No lingering loading state; user can respond immediately after the first question.

5. **Backend Robustness**
   - Continue using bracket counting to ensure only complete JSON objects are parsed and yielded.
   - Log and handle any parse errors gracefully.
   - Success: No parse errors, no stuck frontend.

## Success Criteria
- Initial case intro appears in <10s for most cases.
- Each logical step (intro, question, feedback) is streamed as a separate JSON object.
- Frontend displays each step promptly, removes loading bubble, and shows input box at the right time.
- No parse errors or stuck UI.

## Next Steps
- Benchmark current response times with the new concise prompt.
- Update frontend logic for loading and input box display.
- (Optional) Experiment with more granular streaming if further speedup is needed. 