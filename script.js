document.addEventListener('DOMContentLoaded', () => {

    // --- Copy to Clipboard Logic ---
    const copyBtn = document.getElementById('copyBtn');
    const toast = document.getElementById('copyToast');

    copyBtn.addEventListener('click', () => {
        const codeText = "git clone https://github.com/magpie-lang/magpie.git\ncd magpie\ncargo build -p magpie_cli";

        navigator.clipboard.writeText(codeText).then(() => {
            // Visual feedback
            const originalIcon = copyBtn.innerHTML;
            copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            copyBtn.style.color = "var(--success)";
            copyBtn.style.borderColor = "var(--success)";

            // Show toast
            toast.classList.add('show');

            setTimeout(() => {
                copyBtn.innerHTML = originalIcon;
                copyBtn.style.color = "";
                copyBtn.style.borderColor = "";
                toast.classList.remove('show');
            }, 2000);
        });
    });

    // --- Scroll Animations (Intersection Observer) ---
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');

                // Animate bars if it's a benchmark card
                const bars = entry.target.querySelectorAll('.bar-fill');
                bars.forEach(bar => {
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => {
                        bar.style.width = width;
                    }, 50);
                });

                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.fade-up').forEach(el => {
        observer.observe(el);
    });

    // --- Chatbot Logic ---
    const chatWindow = document.getElementById('chatWindow');
    const suggestionsContainer = document.getElementById('chatSuggestions');
    const suggestionBtns = document.querySelectorAll('.suggestion-btn');

    const responses = {
        "Why is Magpie good for LLMs?": "Magpie relies on explicit SSA (Static Single Assignment) form, predictable naming, and zero hidden control flows. It has a tiny vocabulary ratio, meaning the LLM has to make far fewer choices, leading to practically zero compilation errors.",
        "How fast is the compiler?": "Incredibly fast. Because there is no complex type inference or metaprogramming to resolve, Magpie provides feedback in ~155ms. This tight loop allows agents to fix issues and verify them near-instantly.",
        "What does the code look like?": "It looks like a mix between Rust and LLVM IR. You use strict basic blocks, explicit ownership transfers (borrow/share), and canonical opcodes like `i.add { lhs=%a, rhs=%b }`. No operator overloading, no surprises.",
        "Is there a garbage collector?": "No. Magpie uses a deterministic approach with ARC (Automatic Reference Counting) for heap objects, combined with strict Rust-like borrowing rules for safe mutability without pauses.",
        "Can I run Native binaries?": "Yes! Magpie compiles down to LLVM IR, which is then lowered into highly optimized native machine code (e.g. via Clang), ensuring performance on par with Rust and C++.",
        "How does memory management work?": "It splits responsibilities: ARC manages the lifetimes of heap objects so you don't leak memory, while the Ownership system (shared, borrow, mutborrow, weak) ensures safe aliasing and mutation.",
        "Is the language object-oriented?": "Not in the traditional sense. It has structs and enums (both value and heap variations), and you can define traits (like `hash` or `eq`), but there's no class inheritance or hidden method dispatch.",
        "How does error handling work?": "It uses explicit `TResult<Ok, Err>` and `TOption<T>` types, similar to Rust. There are no invisible exceptions; all error paths must be explicitly handled via explicit branch instructions.",
        "Are there closures or lambdas?": "No. Closures imply a hidden environment layout. Instead, Magpie uses `TCallable`, where you list exactly which values to capture via `callable.capture`. Visibility is key for AI agents.",
        "Why the % and @ symbols?": "Magpie uses `@` to denote global function symbols (like `@main`) and `%` for local SSA values (like `%result`). This makes pattern matching incredibly simple for both the compiler parser and the LLM.",
        "Is Magpie statically typed?": "Yes, absolutely. It uses strong static typing with primitives like `i64` and `bool`, builtin generics like `Array<T>`, and strict ownership modifiers."
    };

    function appendMessage(text, isUser = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;

        if (isUser) {
            msgDiv.innerHTML = `<span class="prompt">user></span> ${text}`;
            chatWindow.insertBefore(msgDiv, suggestionsContainer);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        } else {
            msgDiv.innerHTML = `<span class="prompt">magpie></span> <span class="typing-target"></span>`;
            chatWindow.insertBefore(msgDiv, suggestionsContainer);

            const target = msgDiv.querySelector('.typing-target');
            target.classList.add('typing-cursor');

            typeText(target, text, () => {
                target.classList.remove('typing-cursor');
                // Re-enable buttons if needed, or hide them
                suggestionsContainer.style.display = 'flex';
                chatWindow.scrollTop = chatWindow.scrollHeight;
            });
        }
    }

    function typeText(element, text, callback) {
        let i = 0;
        const speed = 20; // ms per char

        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                chatWindow.scrollTop = chatWindow.scrollHeight;
                setTimeout(type, speed);
            } else {
                if (callback) callback();
            }
        }

        type();
    }

    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const query = e.target.getAttribute('data-query');
            const response = responses[query];

            // Hide suggestions during processing
            suggestionsContainer.style.display = 'none';

            // Add user message
            appendMessage(query, true);

            // Simulate brief thinking delay
            setTimeout(() => {
                appendMessage(response, false);
            }, 500);

            // Remove the clicked button to cycle through questions
            e.target.remove();

            // If no suggestions left, hide container permanently
            if (suggestionsContainer.children.length === 0) {
                // optionally add a final message
            }
        });
    });

});
