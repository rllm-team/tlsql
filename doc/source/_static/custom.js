// ============================================
// MODERNIZED TLSQL DOCUMENTATION JAVASCRIPT
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to code blocks
    addCodeCopyButtons();

    // Original attribute grouping functionality
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
                    font-size: 0.95em !important;
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
                font-size: 0.95em !important;
                font-weight: bold !important;
                font-family: inherit !important;
                font-style: normal !important;
                line-height: inherit !important;
                letter-spacing: normal !important;
                float: none !important;
                clear: none !important;
                margin: 0 !important;
                padding: 0 !important;
                white-space: nowrap !important;
            `;
            
            // Clean up trailing whitespace in dt element to prevent space before colon
            // Remove trailing whitespace from all text nodes, especially the last one
            const walker = document.createTreeWalker(
                dt,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node);
            }
            // Remove trailing whitespace from the last text node
            if (textNodes.length > 0) {
                const lastNode = textNodes[textNodes.length - 1];
                lastNode.textContent = lastNode.textContent.replace(/\s+$/, '');
            }
            
            // Remove headerlink from attribute dt to match tokens page style
            // This ensures all pages have consistent formatting without headerlink
            const headerlink = dt.querySelector('a.headerlink');
            if (headerlink) {
                headerlink.remove();
            }
            
            // Clean up whitespace after removing headerlink
            const sigName = dt.querySelector('.sig-name.descname');
            if (sigName) {
                sigName.style.cssText = `
                    font-weight: bold !important;
                `;
                
                // Remove any trailing whitespace text nodes after sig-name
                let sibling = sigName.nextSibling;
                while (sibling) {
                    const nextSibling = sibling.nextSibling;
                    if (sibling.nodeType === Node.TEXT_NODE && /^\s*$/.test(sibling.textContent)) {
                        dt.removeChild(sibling);
                    }
                    sibling = nextSibling;
                }
            }
            
            if (dds[index]) {
                const dd = dds[index];
                dd.style.cssText = `
                    display: inline !important;
                    font-size: 0.95em !important;
                    font-family: inherit !important;
                    font-style: normal !important;
                    font-weight: normal !important;
                    line-height: inherit !important;
                    letter-spacing: normal !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    float: none !important;
                    clear: none !important;
                `;
                
                const paragraphs = dd.querySelectorAll('p');
                paragraphs.forEach(function(p) {
                    p.style.cssText = `
                        display: inline !important;
                        font-size: 0.95em !important;
                        font-family: inherit !important;
                        font-style: normal !important;
                        font-weight: normal !important;
                        line-height: inherit !important;
                        letter-spacing: normal !important;
                        margin: 0 !important;
                        padding: 0 !important;
                    `;
                });
            }
        });
    });
    
    // No longer need convert function redirection since we merged the functions
    
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

// ============================================
// CODE COPY BUTTON FUNCTIONALITY
// ============================================

function addCodeCopyButtons() {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('pre, .highlight pre, .code-block pre');

    codeBlocks.forEach(function(codeBlock) {
        // Skip if already has a copy button
        if (codeBlock.querySelector('.copy-btn')) return;

        // Create copy button container
        const copyContainer = document.createElement('div');
        copyContainer.className = 'code-copy-container';
        copyContainer.style.cssText = `
            position: relative;
            margin-bottom: 1rem;
        `;

        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn';
        copyButton.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        `;
        copyButton.title = 'Copy code';
        copyButton.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: var(--color-brand-primary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 8px;
            cursor: pointer;
            font-size: 12px;
            opacity: 0;
            transition: all 0.3s ease;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        // Add hover effect to container
        copyContainer.addEventListener('mouseenter', function() {
            copyButton.style.opacity = '1';
        });

        copyContainer.addEventListener('mouseleave', function() {
            copyButton.style.opacity = '0';
        });

        // Add copy functionality
        copyButton.addEventListener('click', function() {
            const code = codeBlock.textContent || codeBlock.innerText;
            navigator.clipboard.writeText(code).then(function() {
                // Show success feedback
                const originalHTML = copyButton.innerHTML;
                copyButton.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20,6 9,17 4,12"></polyline>
                    </svg>
                `;
                copyButton.style.background = 'var(--color-brand-secondary)';

                setTimeout(function() {
                    copyButton.innerHTML = originalHTML;
                    copyButton.style.background = 'var(--color-brand-primary)';
                }, 2000);
            }).catch(function(err) {
                console.error('Failed to copy: ', err);
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = code;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);

                // Show success feedback
                const originalHTML = copyButton.innerHTML;
                copyButton.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20,6 9,17 4,12"></polyline>
                    </svg>
                `;
                copyButton.style.background = 'var(--color-brand-secondary)';

                setTimeout(function() {
                    copyButton.innerHTML = originalHTML;
                    copyButton.style.background = 'var(--color-brand-primary)';
                }, 2000);
            });
        });

        // Wrap code block
        const parent = codeBlock.parentElement;
        parent.insertBefore(copyContainer, codeBlock);
        copyContainer.appendChild(codeBlock);
        copyContainer.appendChild(copyButton);

        // Ensure code block has proper styling
        codeBlock.style.cssText = `
            margin: 0;
            padding: 1rem;
            background: var(--color-code-background);
            border-radius: 8px;
            overflow-x: auto;
        `;
    });
}

// ============================================
// ENHANCED INTERACTIVE ELEMENTS
// ============================================

// Add smooth scrolling for anchor links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Add back-to-top button
const backToTopBtn = document.createElement('button');
backToTopBtn.innerHTML = '↑';
backToTopBtn.className = 'back-to-top-btn';
backToTopBtn.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: var(--tlsql-gradient);
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    cursor: pointer;
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
`;

backToTopBtn.addEventListener('click', function() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// Show/hide back-to-top button based on scroll position
window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) {
        backToTopBtn.style.opacity = '1';
    } else {
        backToTopBtn.style.opacity = '0';
    }
});

document.body.appendChild(backToTopBtn);
