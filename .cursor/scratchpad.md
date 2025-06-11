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
- [x] Add condition_stats to /progress endpoint response

## Current Status / Progress Tracking
- Status: Streaming fix for 'is_completed' in /continue_case implemented
- Completed: Now, 'is_completed' is only included in the response when the case is truly finished (i.e., after '[CASE COMPLETED]' is detected in the final message). For all intermediate steps, it is omitted.
- Next: Test the streaming endpoint with a real case to ensure the frontend only shows feedback and post-case actions at the correct time.
- The /progress endpoint now includes a condition_stats field with per-condition total_cases and avg_score, as required by the frontend. Implementation used the already-fetched completed_cases list for efficient aggregation.
- Awaiting user confirmation after frontend testing.

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
- For per-user stats aggregation (ward, condition), use the already-fetched completed_cases list to avoid extra DB queries and improve performance.

# Medical Chatbot Protocol Redesign (Planner Mode)

## Background and Motivation
- The current backend streams every token as a separate message, and the frontend cannot reliably detect when the assistant has finished a turn, leading to UI/UX issues (e.g., input box not appearing at the right time).
- The goal is a robust, stepwise chatbot for medical cases, with clear turn-taking, flexible prompt management, and reliable feedback/score at the end.

## Key Challenges and Analysis
- **Turn Boundaries:** Need a clear signal from backend to frontend for when the assistant has finished a turn (i.e., user can respond).
- **Streaming:** Want to preserve streaming for responsiveness, but must aggregate assistant output per turn.
- **Prompt Flexibility:** System prompt should be easily changeable (env/config/admin endpoint).
- **Case Completion:** Need a reliable marker for end-of-case, with feedback and score.
- **Frontend/Backend Contract:** Both must agree on message structure and turn protocol.

## High-level Task Breakdown
- [ ] Define a message protocol for backend-to-frontend communication (including turn boundaries and case completion).
- [ ] Update backend to aggregate assistant output per turn and send a `turn_complete` signal after each assistant message.
- [ ] Update frontend to display streamed assistant messages and enable input only after `turn_complete` is received.
- [ ] Ensure backend sends a structured `[CASE COMPLETED]` marker and feedback/score JSON at the end of the case.
- [ ] Make system prompt configurable (env var, config file, or admin endpoint).
- [ ] Test the full flow for various cases and edge scenarios.

## Project Status Board
- [ ] **Task 1:** Define protocol and message structure
- [ ] **Task 2:** Backend: Aggregate assistant output and send `turn_complete`
- [ ] **Task 3:** Frontend: Listen for `turn_complete` and manage input box
- [ ] **Task 4:** Backend: Configurable system prompt
- [ ] **Task 5:** End-to-end testing and validation

## Current Status / Progress Tracking
- Planner mode active. Protocol and requirements being defined.

## Lessons
- Streaming every token is not enough; need explicit turn boundaries for reliable chat UX.
- Both backend and frontend must follow a shared protocol for turn-taking and case completion.

## Note on System Prompt
- For this iteration, the system prompt will be returned in the chat for manual update in the OpenAI Assistant Playground. No code/config change for prompt management is needed at this stage.

## Next Executor Steps
- [ ] Backend: Implement streaming protocol with `turn_complete` and `case_completed` messages.
- [ ] Frontend: Update to listen for these messages and manage input/feedback display accordingly.
- [ ] Return the current recommended system prompt in the chat for manual use. 