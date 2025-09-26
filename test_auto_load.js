// Simple script to test auto load functionality
async function testAutoLoad() {
    console.log('Testing auto-load...');

    try {
        const healthResponse = await fetch('http://localhost:8004/health/neo4j');
        const health = await healthResponse.json();
        console.log('Health check result:', health);

        if (health.status === 'ok' && health.nodeCount > 0) {
            console.log('Neo4j is healthy and has data, attempting to load KG...');

            // Load KG using FormData like the frontend does
            const formData = new FormData();
            formData.append('uri', 'bolt://localhost:7687');
            formData.append('user', 'neo4j');
            formData.append('password', '');
            formData.append('limit', '1000');
            formData.append('sample_mode', 'false');
            formData.append('load_complete', 'false');

            const loadResponse = await fetch('http://localhost:8004/load_kg_from_neo4j', {
                method: 'POST',
                body: formData
            });

            if (loadResponse.ok) {
                const result = await loadResponse.json();
                console.log('KG Load successful:', result);
                console.log('Nodes count:', result.graph_data?.nodes?.length);
                console.log('Relationships count:', result.graph_data?.relationships?.length);
            } else {
                console.error('KG Load failed:', loadResponse.status);
                const errorText = await loadResponse.text();
                console.error('Error text:', errorText);
            }
        } else {
            console.log('Neo4j not ready for auto-load');
        }
    } catch (error) {
        console.error('Auto-load test error:', error);
    }
}

testAutoLoad();
