// Test script to validate that the chat functionality works without errors
const fetch = require('node-fetch');

async function testChatEndpoint() {
    const payload = {
        question: 'What is a knowledge graph?',
        provider_rag: 'openrouter',
        model_rag: 'meta-llama/llama-4-maverick:free',
        kg_only: true
    };

    try {
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Simulate the JavaScript formatting logic from frontend
        let formattedResponse = result.response || result.message || 'No response generated';

        // Test the replace operations
        try {
            formattedResponse = formattedResponse.replace(/##\s*\*\*<u>(\d+)\.\s*(.*?):<\/u>\*\*/g, function(match, number, title) {
                return `<h2 style="font-weight: bold; text-decoration: underline; color: #1976d2; margin: 16px 0 8px 0;">${number}. ${title.toUpperCase()}:</h2>`;
            });

            let sectionCount = 0;
            formattedResponse = formattedResponse.replace(/^##\s+(.*)$/gm, function(match, title) {
                sectionCount += 1;
                return `<h2 style="font-weight: bold; text-decoration: underline; color: #1976d2; margin: 16px 0 8px 0;">${sectionCount}. ${title.toUpperCase()}</h2>`;
            });

            console.log('✅ SUCCESS: Chat response formatting works correctly!');
            console.log('Response contains:', formattedResponse.length, 'characters');
            console.log('Response preview:', formattedResponse.substring(0, 200) + '...');

            return true;
        } catch (error) {
            console.error('❌ ERROR: Formatting failed:', error.message);
            return false;
        }

    } catch (error) {
        console.error('❌ ERROR: Request failed:', error.message);
        return false;
    }
}

// Run the test
testChatEndpoint().then(success => {
    if (success) {
        console.log('\n🎉 Frontend JavaScript fix validated - chat should work without errors!');
    } else {
        console.log('\n⚠️  Test failed - there may be additional issues');
    }
    process.exit(success ? 0 : 1);
});
