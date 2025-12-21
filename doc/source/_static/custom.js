// JavaScript to group all attributes into a single box
// This runs after page load to fix attribute display

document.addEventListener('DOMContentLoaded', function() {
    // Find all attribute description lists that are NOT already in a container
    const allAttributeLists = document.querySelectorAll('dl.py.attribute');
    const attributeLists = Array.from(allAttributeLists).filter(function(dl) {
        // Skip if already inside an attributes-container
        return !dl.closest('.attributes-container');
    });
    
    if (attributeLists.length === 0) return;
    
    console.log('Found', attributeLists.length, 'unwrapped attribute lists');
    
    // Group attributes by their immediate parent container
    const attributeGroups = new Map();
    
    attributeLists.forEach(function(dl) {
        // Find the immediate parent (usually a dd or div)
        const parent = dl.parentElement;
        if (parent) {
            if (!attributeGroups.has(parent)) {
                attributeGroups.set(parent, []);
            }
            attributeGroups.get(parent).push(dl);
        }
    });
    
    // Process each group of attributes
    attributeGroups.forEach(function(attributes, parentContainer) {
        if (attributes.length > 1) {
            // Multiple attributes in same container - wrap them
            // Check if parentContainer is already an attributes-container
            if (parentContainer.classList.contains('attributes-container')) {
                // Already wrapped, just ensure attributes are styled correctly
                attributes.forEach(function(dl) {
                    dl.style.cssText = `
                        margin-top: 0 !important;
                        margin-bottom: 0.25em !important;
                        padding: 0 !important;
                        background-color: transparent !important;
                        border: none !important;
                        border-radius: 0 !important;
                    `;
                    dl.classList.add('in-container');
                });
                return;
            }
            
            // Create wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'attributes-container';
            wrapper.style.cssText = `
                margin-top: 1em !important;
                margin-bottom: 1em !important;
                padding-top: 0.75em !important;
                padding-bottom: 0.75em !important;
                padding-left: 0.75em !important;
                padding-right: 0.75em !important;
                background-color: #f8f9fa !important;
                border-left: 3px solid #0066cc !important;
                border-radius: 4px !important;
            `;
            
            // Add "Attributes" title (only if not already exists in parent)
            if (!parentContainer.querySelector('.attributes-title')) {
                const title = document.createElement('div');
                title.className = 'attributes-title';
                title.textContent = 'ATTRIBUTES';
                title.style.cssText = `
                    font-weight: 600 !important;
                    font-size: 0.9em !important;
                    color: #0066cc !important;
                    margin-bottom: 0.5em !important;
                    text-transform: uppercase !important;
                    letter-spacing: 0.5px !important;
                `;
                wrapper.appendChild(title);
            }
            
            // Insert wrapper before first attribute
            parentContainer.insertBefore(wrapper, attributes[0]);
            
            // Move all attributes into wrapper and remove their individual styles
            attributes.forEach(function(dl) {
                // Remove all individual box styles
                dl.style.cssText = `
                    margin-top: 0 !important;
                    margin-bottom: 0.25em !important;
                    padding: 0 !important;
                    background-color: transparent !important;
                    border: none !important;
                    border-radius: 0 !important;
                `;
                
                // Mark as in container
                dl.classList.add('in-container');
                
                wrapper.appendChild(dl);
            });
            
            console.log('Wrapped', attributes.length, 'attributes in container');
        } else if (attributes.length === 1) {
            // Single attribute - still wrap it for consistency
            const dl = attributes[0];
            const parent = dl.parentElement;
            
            // Check if parent is already a container
            if (parent.classList.contains('attributes-container')) {
                return;
            }
            
            const wrapper = document.createElement('div');
            wrapper.className = 'attributes-container';
            wrapper.style.cssText = `
                margin-top: 1em !important;
                margin-bottom: 1em !important;
                padding-top: 0.75em !important;
                padding-bottom: 0.75em !important;
                padding-left: 0.75em !important;
                padding-right: 0.75em !important;
                background-color: #f8f9fa !important;
                border-left: 3px solid #0066cc !important;
                border-radius: 4px !important;
            `;
            
            const title = document.createElement('div');
            title.className = 'attributes-title';
            title.textContent = 'ATTRIBUTES';
            title.style.cssText = `
                font-weight: 600 !important;
                font-size: 0.9em !important;
                color: #0066cc !important;
                margin-bottom: 0.5em !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
            `;
            wrapper.appendChild(title);
            
            parent.insertBefore(wrapper, dl);
            
            dl.style.cssText = `
                margin-top: 0 !important;
                margin-bottom: 0.25em !important;
                padding: 0 !important;
                background-color: transparent !important;
                border: none !important;
                border-radius: 0 !important;
            `;
            dl.classList.add('in-container');
            
            wrapper.appendChild(dl);
        }
    });
    
    // Ensure all dt/dd are inline for all attributes (including those already in containers)
    document.querySelectorAll('dl.py.attribute').forEach(function(dl) {
        const dts = dl.querySelectorAll('dt');
        const dds = dl.querySelectorAll('dd');
        
        dts.forEach(function(dt, index) {
            dt.style.cssText = `
                display: inline !important;
                font-weight: bold !important;
                float: none !important;
                clear: none !important;
                margin: 0 !important;
                padding: 0 !important;
            `;
            
            // Extract text content from nested spans
            const sigName = dt.querySelector('.sig-name.descname');
            if (sigName) {
                sigName.style.cssText = `
                    font-weight: bold !important;
                `;
            }
            
            if (dds[index]) {
                const dd = dds[index];
                dd.style.cssText = `
                    display: inline !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    float: none !important;
                    clear: none !important;
                `;
                
                const paragraphs = dd.querySelectorAll('p');
                paragraphs.forEach(function(p) {
                    p.style.cssText = `
                        display: inline !important;
                        margin: 0 !important;
                        padding: 0 !important;
                    `;
                });
            }
        });
    });
    
    // Dark mode support
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        const containers = document.querySelectorAll('.attributes-container');
        containers.forEach(function(container) {
            container.style.backgroundColor = '#242424';
            container.style.borderLeftColor = '#4da6ff';
            const title = container.querySelector('.attributes-title');
            if (title) {
                title.style.color = '#4da6ff';
            }
        });
    }
});
