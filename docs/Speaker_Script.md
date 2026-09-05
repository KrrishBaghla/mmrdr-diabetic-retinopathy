# Speaker Script — Diabetic Retinopathy Presentation
### Your slide-by-slide guide to sound confident, clear, and impressive

**How to use this document:** Don't memorize it word-for-word — read it through 2-3 times so the *ideas* stick, then speak in your own natural voice. The goal is to sound like you're explaining your work to someone smart but busy, not reciting a report. Where you see **🎯 Delivery tip**, that's a small acting note — a pause, a gesture, a change of pace. Those little moments are what make a talk feel confident instead of rehearsed.

One more thing before you start: **slow down more than feels natural.** Everyone's biggest mistake in a big presentation is rushing. You know this material better than anyone in the room — you have permission to take your time.

---

## Slide 1 — Title

**What to say:**
"Good morning everyone. Thank you for the time today. My project is called *Multi-Task Deep Learning for Diabetic Retinopathy* — and in simple terms, it's about teaching a computer to look at a photo of the back of someone's eye and tell you two things at once: how severe their diabetic eye disease is, and exactly what damage is causing it. I ran ten experiments over this internship, and today I'll walk you through what I built, what worked, what didn't, and what I learned along the way."

**🎯 Delivery tip:** Don't start talking immediately when the slide appears. Let it sit for two seconds, take a breath, then begin. This single pause makes you look completely in control of the room.

---

## Slide 2 — Clinical Motivation

**What to say:**
"Let's start with why this matters. Diabetic retinopathy is damage to the retina caused by diabetes, and it's one of the biggest causes of blindness that could actually be *prevented* — if it's caught early enough. The problem is catching it early. Right now, that requires a trained eye doctor to look at a photo and manually decide how bad it is and what's wrong. But there simply aren't enough eye doctors in the world to screen every diabetic patient, especially in places with limited healthcare access. So the real opportunity here is: can a computer do that first pass — reliably, consistently, and at scale — so doctors can focus their time on the patients who need them most?"

**🎯 Delivery tip:** This is your "why should anyone care" slide. Say it like you mean it — a little slower than the rest of your talk.

---

## Slide 3 — Problem Statement & Objectives

**What to say:**
"Here's the core idea behind everything I built. Every eye image in this project gets scored in two ways. First, a severity grade from 0 to 4 — 0 means no disease, 4 means the most severe stage. Second, a list of exactly which of seven specific types of damage are visible — things like small bleeding spots or abnormal blood vessels. Now here's the key insight: these two things aren't separate problems. The damage you see is *literally what decides* the severity grade. So instead of building two different systems, I built one model that learns both jobs together, from the same image, at the same time. My goals for this internship were simple: build strong versions of this model on two different types of eye photos, then dig deep into the hardest part of the problem — catching the rare, most dangerous types of damage — and be completely honest about what worked and what didn't."

**🎯 Delivery tip:** Point at the color scale (green to red) when you mention severity grades, and at the seven boxes when you mention lesion types. Physically guiding their eyes keeps them engaged.

---

## Slide 4 — Dataset (MMRDR)

**What to say:**
"This is the dataset I worked with, called MMRDR. It gives us eye images in three different formats, but I worked with two of them. On the left is a standard fundus photo — this is what most eye clinics already use, a regular camera pointed at the retina. On the right is something called ultra-widefield imaging, which captures a much, much wider view of the retina — almost the entire inside surface of the eye instead of just the center. That matters because some of the most dangerous damage happens right at the edges, out in the periphery, where a normal camera simply can't see. I had about eleven thousand of the standard photos and just over ten thousand of the wide-field ones, and every single image comes labeled with both the severity grade and the damage list I just described."

**🎯 Delivery tip:** Gesture from the small clean circle (left photo) to the wider, more dramatic-looking image (right) as you explain the field-of-view difference — the visual contrast does half the explaining for you.

---

## Slide 5 — Multi-Task Learning Formulation

**What to say:**
"So here's exactly how the model is built, in plain terms. An eye image goes in on the left. It passes through what I call a shared backbone — think of this as the part of the model that actually *looks at* the image and extracts useful visual information, like edges, textures, and shapes relevant to the retina. Then that same visual understanding splits into two separate output paths: one path predicts the severity grade, the other predicts which of the seven damage types are present. Both paths are trained together, from the same shared understanding of the image. That's the whole idea in one sentence: one shared brain, two separate answers. And I want you to remember this diagram, because every single one of my ten experiments is really just a variation on this same basic structure — what changes each time is which specific pieces get modified."

**🎯 Delivery tip:** Trace the arrows with your hand, left to right, exactly once, slowly. Don't rush through the diagram — this is the mental model the rest of your talk depends on.

---

## Slide 6 — Methodology Pipeline & Metrics

**What to say:**
"Before the model ever sees an image, a few things happen first. We clean up the photo — cropping out the black borders and any camera artifacts so the model only looks at real eye tissue. Then it goes through the backbone I just showed you, splits into the two heads, gets scored by a loss function that tells the model how wrong it was, and finally — and this part is important — I use a technique called Grad-CAM, which creates a heat map showing exactly *where* in the image the model was looking when it made its decision. This is how I keep myself honest: it lets me check whether the model is actually looking at real disease, like blood vessels and bleeding spots, instead of some random shortcut. As for how I judge success: I look at plain accuracy on the severity grade, a more forgiving score that gives partial credit for being close, and — most importantly for the second half of this talk — how well the model catches the three rarest, most dangerous types of damage specifically."

**🎯 Delivery tip:** When you say "Grad-CAM," pause slightly — it's a term some in the audience may not know, and giving it a beat lets it land before you explain it.

---

## Slide 7 — Experimental Roadmap

**What to say:**
"Here's the map for the rest of the talk, so you always know where we are. I ran ten experiments in two phases. The first three experiments — the top row here — are about building solid, working baseline models on each type of eye photo. Nothing fancy yet, just: does this basic approach work well? The second phase, the bottom seven experiments, all live on the wide-field images, and every single one of them is attacking the exact same hard problem from a different angle: how do we make the model much better at catching the rare, sight-threatening damage that defines the worst stage of this disease? Loss functions, feature learning, smarter training data sampling, architecture choices, even generating brand new synthetic images — six completely different strategies, all aimed at the same target. Keep this slide in your head; I'll come back to this structure throughout."

**🎯 Delivery tip:** This is a great slide to briefly step back from the screen and make eye contact with the room — it signals "here's the shape of my whole talk," which builds trust early.

---

## Slide 8 — Experiment I: Architecture

**What to say:**
"My first experiment, and I like to explain this one with a simple picture: imagine you're looking at a crowded photo. If you zoom in close, you can read fine details, but you lose the big picture. If you zoom out, you see the whole scene, but you lose the small details. A single-scale model has exactly that trade-off — and disease markers like tiny bleeding spots are small, while grading also needs the big picture, like where the optic disc is. So I gave the model two eyes instead of one: one branch that keeps the fine, zoomed-in detail from an early layer of the network, and another branch that keeps the big-picture, zoomed-out understanding from a deep layer. I combine both views before the model makes its final decision. That's the whole trick — fine detail and global context, working together instead of trading off against each other."

**🎯 Delivery tip:** Use your two hands as the "two eyes" — one hand close to your face (zoomed in), one hand far out (zoomed out) — then bring them together when you say "combined." Physical gestures make abstract architecture concepts stick.

---

## Slide 9 — Experiment I: Results

**What to say:**
"And here's how it performed. The training went smoothly — the error kept dropping in a clean, stable curve, which tells us the model wasn't struggling or unstable during learning. On the severity grade, it reached just under eighty-three percent accuracy, and on detecting the damage types, over ninety-five percent — though I'll be honest, that second number is a little generous, since most patients simply don't have most types of damage, so predicting 'absent' correctly is somewhat easy. What I actually trust more is the heat map on the right — it shows the model is looking at the optic disc, the blood vessels, and the actual damage spots, not some random unrelated part of the image. That tells me the model is learning real medical signal, not cheating. This becomes our reference point — the benchmark every later experiment gets compared against."

**🎯 Delivery tip:** When you mention the heat map, actually point to the red/hot region in the image and say "right there" — it's a small moment but it makes the evidence feel concrete rather than abstract.

---

## Slide 10 — Experiment II: RETFound Architecture

**What to say:**
"For my second experiment, I asked a different question: what if, instead of a general-purpose image model, I use one that was already pre-trained specifically on eye images? This model is called RETFound. Think of it like hiring someone who already has years of experience specifically with retinas, rather than a generalist. Because it already understands eye anatomy, I didn't need to retrain the whole thing from scratch — I froze most of it, shown in blue here, and only fine-tuned the last small portion, shown in coral, plus the two output heads. This is a much gentler, more targeted way of adapting an already-knowledgeable model to a new specific task."

**🎯 Delivery tip:** The "hiring an experienced specialist" analogy is your anchor line here — say it clearly and pause after it, since it's doing the heavy lifting of the explanation.

---

## Slide 11 — Experiment II: Results & Comparison

**What to say:**
"So did the specialist model win? Actually, no — it came in lower, at about sixty-six and a half percent, compared to almost eighty-three percent for our first model. But I want to be very clear: this isn't a failure of the idea. Look at this table — there are two real reasons for the gap. First, this model only accepts images at a much lower resolution, which means fine details — exactly the kind of tiny damage spots we care about — get lost before the model even sees them. Second, we only fine-tuned a small slice of the network. So this isn't 'foundation models don't work' — it's 'under this exact, resource-limited setup, a fully-trained custom model still wins.' That's a useful, honest finding, and it points directly to what I'd try next: higher resolution, deeper fine-tuning."

**🎯 Delivery tip:** Say "I want to be very clear" with genuine confidence, not defensiveness — this is you demonstrating good scientific judgment, not making an excuse. Professors respond very well to this kind of honesty.

---

## Slide 12 — Experiment III: FPN + Attention

**What to say:**
"Now we move to the wide-field images. Remember, these capture almost the whole retina, including the edges where the most dangerous damage often hides. But that wider view comes with its own headaches — eyelashes and eyelids often creep into the frame. So I built a model with a feature pyramid, which combines information from multiple zoom levels at once, plus a spatial attention module — which is really just a mechanism that lets the model learn to focus on the parts of the image that actually matter, and ignore the rest. Looking at the heat maps here, you can see the attention concentrating right on the blood vessels and the peripheral retina — not on the eyelash artifacts at the edges, which is exactly what we want. This model reached about seventy-four and a half percent grading accuracy — lower than the standard photo model, which makes sense, because this is simply a harder, messier type of image to work with."

**🎯 Delivery tip:** Acknowledge the lower number confidently — frame it as "this makes sense" rather than apologizing for it. Context turns a lower number into an insight instead of a weak spot.

---

## Slide 13 — Baseline Synthesis

**What to say:**
"So let's put all three baselines side by side. The fully-trained custom model on standard photos wins clearly, at almost eighty-three percent. The specialist foundation model comes in lowest, for the resolution reasons we just discussed. And the wide-field model sits in the middle — lower than the standard photo model, but remember, it's the only one of the three that can actually *see* the peripheral damage that matters most clinically. The big lesson from these three experiments: how much of the model you actually fine-tune, and what resolution you feed it, matters just as much as which fancy pretrained model you start with. And with that, we're done with baselines — every experiment from here forward focuses entirely on the wide-field images and one very specific, very hard problem."

**🎯 Delivery tip:** This is a natural "chapter break" in your talk. Take a breath, maybe take a sip of water if you have one nearby — it signals to the audience that we're moving into a new part of the story.

---

## Slide 14 — The Rare-Lesion Challenge (Transition)

**What to say:**
"I want to slow down on this slide, because everything for the rest of the talk builds on it. These three types of damage — neovascularization, vitreous hemorrhage, and retinal detachment — are the markers of the most severe, sight-threatening stage of this disease. And here's the problem: they're incredibly rare in the data. Retinal detachment shows up in only two-point-three percent of our wide-field images — just two hundred and thirty-eight images out of over ten thousand. But here's the clinical reality that makes this so important: if the model misses one of these three and tells a patient they're fine, that patient could lose their vision permanently. If it misses a common, harmless spot, almost nothing bad happens. So from this point forward, every single experiment I'm about to show you is a different technical attempt to solve exactly this one problem: how do you make a model highly sensitive to something it barely ever sees in training?"

**🎯 Delivery tip:** Slow way down on "two hundred and thirty-eight images out of over ten thousand." Numbers like this are what make a room go quiet — let it land instead of rushing past it.

---

## Slide 15 — Experiment IV: Clinically-Weighted Loss

**What to say:**
"My first attempt at solving this: change how much each type of mistake 'costs' the model during training. Normally, you'd weight things purely by how rare they are statistically. But I went one step further and combined rarity with actual clinical risk — I told the model, in effect, 'a missed retinal detachment should hurt you much more than a missed common spot, because a doctor would care about it much more too.' Look at the table: recall on the three high-risk types jumped up nicely, while recall on the common, low-risk types dropped a lot. And I want to be upfront — this drop isn't a bug, it's exactly what I designed it to do. For a screening tool, it's far better to over-flag common, harmless things than to miss something dangerous. That's a trade-off any doctor would tell you is the right one to make."

**🎯 Delivery tip:** When you say "this drop isn't a bug, it's exactly what I designed it to do," say it with a small smile — it shows you're in control of the trade-off, not surprised by it.

---

## Slide 16 — Experiment V: Asymmetric FN-Penalised Loss

**What to say:**
"This experiment attacks the same problem from a slightly different angle. Instead of just weighting classes by risk, I changed the *shape* of the mistake penalty itself. Here's the reasoning: telling a patient they're fine when they actually have a dangerous condition — a false negative — is a completely different kind of mistake than falsely flagging something that isn't there. One can cost someone their eyesight; the other just means an extra check-up. So I built a loss function that punishes missing the three high-risk conditions roughly six times harder than a false alarm. The result: strong, stable grading — most of the model's mistakes were off by just one severity level, not wildly wrong — plus reliable lesion detection, and critically, it didn't crush the common lesion detection nearly as hard as my first attempt did. This is a genuinely better trade-off."

**🎯 Delivery tip:** The phrase "six times harder" is a great concrete number to slow down on — specific numbers like this make technical explanations feel credible rather than hand-wavy.

---

## Slide 17 — Experiment VI: Supervised Contrastive Learning

**What to say:**
"Here I tried something completely different: instead of touching the loss weighting again, I changed what the model actually *learns to see*. The idea is this — re-weighting a loss only adjusts where the model draws its decision line, but if the underlying visual features for rare and common cases are still tangled together in the model's 'mental map,' that decision line is fighting an uphill battle. So I added a technique that actively pulls the visual representations of all rare, dangerous cases closer together, and pushes them away from the normal cases — literally reshaping how the model organizes what it sees internally. This gave us the best detection score of any model so far in this thread, while keeping grading just as strong. It tells us something valuable: fixing the *loss function* and fixing the *internal representation* are two genuinely different levers, and they can work together."

**🎯 Delivery tip:** The phrase "mental map" is doing a lot of work here — say it slowly and let the metaphor sit before moving on.

---

## Slide 18 — Experiment VII: Learned Uncertainty Weighting

**What to say:**
"So far, I've been hand-picking how much to weight each part of the training — grading versus lesion detection versus this new representation-learning piece. This experiment asks: can the model figure out that balance on its own? I gave each task a learnable 'confidence' parameter, so the model can automatically lean more heavily into tasks it's doing well on. Now, here's something I want to be fully transparent about — and I think this honesty matters. Look at this weight chart: the lesion task's weight shot straight up and hit its maximum allowed value almost immediately. So in practice, this didn't behave like a smart, adaptive balance — it behaved more like the model deciding, very early on, to prioritize lesion detection extremely heavily and then staying there. That said, this gave us the single best rare-lesion detection result of the entire study. I'd frame it honestly: this shows us what's *possible* when the model prioritizes lesion detection strongly, more than it proves the 'learned adaptivity' idea worked exactly as intended."

**🎯 Delivery tip:** This slide is your best chance to impress professors specifically — most students would hide this caveat. Say it calmly and directly; it's a strength of your work, not a weakness. Make eye contact with the room here.

---

## Slide 19 — Synthesis: Loss & Representation (IV–VII)

**What to say:**
"Let's line up these four experiments together and see the pattern. As we move from simple risk-weighting, to a smarter loss shape, to reshaping the internal representation, to learned prioritization — the rare-lesion numbers keep climbing, with experiment seven coming out on top across the board. But I want to add an important nuance here: this isn't a perfectly clean, isolated comparison, because each experiment builds directly on top of the previous one's network — experiment six adds onto experiment five, and experiment seven adds onto experiment six. So the improvements are cumulative, and experiment seven's win specifically reflects that saturated, heavily-prioritized weighting we just discussed, not a perfectly controlled test. I think that nuance is important for anyone who wants to build on this work."

**🎯 Delivery tip:** Use your hand to trace a rising staircase motion across the four rows of the table as you say "keep climbing" — a small physical gesture that reinforces the trend visually.

---

## Slide 20 — Experiment VIII: Curriculum Resampling

**What to say:**
"This experiment takes a completely different approach: instead of touching the loss function at all, I changed *which images the model sees more often* during training. I kept the network and the loss function exactly the same as experiment six, and only changed the sampling strategy. The system tracks, epoch by epoch, which rare classes the model is currently struggling to recall, and gradually shows it more of exactly those images, ramping this up slowly after a short warm-up period. I also want to flag something good here — this is the first experiment in this whole thread to use a properly separated, balanced train-and-validation split, which is a more rigorous setup than what I used earlier. The results were striking: recall on the rarest, most dangerous condition reached a perfect one hundred percent, and grading quality actually reached its highest point in the entire study at the same time — so fixing the sampling didn't cost us anything on the grading side."

**🎯 Delivery tip:** "A perfect one hundred percent" is a genuinely exciting number — let your energy rise slightly here, it's a strong result and it's fine to sound a little pleased about it.

---

## Slide 21 — Experiment IX: Head Architecture Ablation

**What to say:**
"This next one is my favorite experiment to talk about, because it's a negative result — and I think negative results, done properly, are just as valuable as positive ones. The question was: if I give the rare, dangerous conditions their own dedicated output branch, completely separate from the common ones, will the model detect them better? It's a very reasonable hypothesis. To test it fairly, I built three versions of the model with carefully matched parameter counts, so that any difference we see is really coming from the separation itself, and not just from one version simply having more capacity. I evaluated all three on a genuine, untouched test set the model had never influenced its own training choices from. And the answer was clearly no — the dedicated branch didn't help, and it actually performed slightly worse than the simpler, shared design. What this tells us is valuable: for this model, the bottleneck isn't how the output layer is organized — it's something deeper, in the core features or the amount of rare training data available."

**🎯 Delivery tip:** Say "I think negative results, done properly, are just as valuable as positive ones" and mean it — this single sentence often earns more respect from a room of researchers than any positive result would.

---

## Slide 22 — Experiment X: Synthetic Diffusion Generation

**What to say:**
"My last experiment takes the most creative approach: rather than reusing the same handful of rare images over and over, what if we generate brand new ones? I built a generative model — the same family of technology behind modern AI image generators — trained specifically on our rare, dangerous cases, so it learns to produce new, varied examples of them. As a first proof of concept, I trained it at a small image size, and it converged nicely and produces genuinely varied outputs — you can see in the comparison here that it captures the overall look, color, and shape of a real retina quite well. But at this small size, it doesn't yet render fine, sharp disease detail convincingly — which is exactly why the natural next step is a higher-resolution version. And I want to be completely upfront: the real question — does adding these generated images actually help a model detect more rare cases in the real world — has not been answered yet. That's the next experiment, not a result I'm claiming today."

**🎯 Delivery tip:** End this one on "that's the next experiment, not a result I'm claiming today" with a calm, matter-of-fact tone — it shows scientific discipline, which lands very well with a faculty audience.

---

## Slide 23 — Master Comparison Table

**What to say:**
"Let's zoom out and look at all ten experiments together in one place. A few things jump out immediately: our very first experiment remains the strongest pure grading model of the whole study. Experiments five through seven show a clear, step-by-step improvement in rare-lesion detection. And experiment seven stands out with the best detection scores overall. I do want to flag, one more time, that experiments eight and nine used a different, more rigorous data split than the others, so their numbers shouldn't be read as a perfectly direct, apples-to-apples comparison with the rest — I've marked those clearly with an asterisk."

**🎯 Delivery tip:** Let this slide breathe — resist the urge to explain every single row. Let the audience's eyes scan the table for a few seconds before you speak; a well-designed table is allowed to do some of the talking for you.

---

## Slide 24 — Key Findings & Research Contributions

**What to say:**
"Let me step back from the individual numbers and tell you what this internship actually contributes, as a whole. First: a fully-trained, purpose-built model beat a specialized pretrained one — but only under a specific, resource-limited setup, which is a nuanced finding, not a blanket statement against pretrained models. Second: I systematically tested six genuinely different technical strategies for the exact same hard problem, all under one consistent, fair evaluation — that breadth of comparison is really the core contribution of this work. Third: the best rare-lesion result came from combining smarter feature learning with strong task prioritization, while a dedicated architecture change surprisingly didn't help at all — a genuinely useful thing for anyone else working on this exact problem to know. And finally, across every single model I trained, the heat-map checks confirmed the model was consistently looking at real, clinically meaningful evidence — which gives me real confidence these results reflect genuine learning, not shortcuts."

**🎯 Delivery tip:** This is your "hire me" slide, in a sense. Slow down, look up from your notes, and deliver this one directly to the room rather than to the screen.

---

## Slide 25 — Limitations & Honest Caveats

**What to say:**
"I want to be fully transparent about the limits of this work, because I believe that's part of doing this properly. Most of my experiments used a random data split rather than a carefully balanced one, so rare cases might not be evenly spread between training and testing. In most experiments, my validation set doubled as my final test set, which isn't ideal — only my last two experiments fixed that properly. Nearly everything here is a single run rather than repeated multiple times, so I can't yet report how much the results might vary by chance. And my very last experiment only shows that the image generator itself works — it doesn't yet prove that using those generated images actually improves real-world detection."

**🎯 Delivery tip:** Deliver this slide with the same steady, confident tone as your results — never apologetic. Limitations presented calmly read as expertise; limitations presented nervously read as insecurity.

---

## Slide 26 — Future Work

**What to say:**
"So where would I take this next? The single biggest priority is re-running the key comparisons multiple times with different random starting points, so I can report real statistical confidence instead of single numbers. I'd also want to revisit the pretrained specialist model with a higher resolution and deeper fine-tuning, to see if it can actually close the gap with the custom model. On the clinical side, the risk weightings I used were something I set by hand — working with an actual clinician to calibrate those properly would make that experiment much more rigorous. And of course, finishing that last experiment — actually measuring whether generated images improve real-world detection — is a clear next step. Longer term, I'd love to explore training one single model across all three imaging types this dataset offers, including the one I didn't get to touch yet."

**🎯 Delivery tip:** Future work slides can feel like a formality — don't let your energy drop here. Say this with the same enthusiasm as your results; it shows you're already thinking past this internship.

---

## Slide 27 — Conclusion

**What to say:**
"To bring it all together: over this internship, I built and tested ten multi-task models across two types of eye imaging, establishing strong baseline performance and then systematically tackling the much harder problem of catching rare, sight-threatening damage from six independent angles. The best result for that hard problem came from combining smarter feature learning with strong prioritization, while a dedicated architecture change turned out not to help — a clean, honest, useful finding either way. Across every model, the heat-map checks confirmed real, clinically grounded learning. And if I had to summarize the actual contribution of this work in one sentence, it's this: a careful, honestly-limited comparison of six genuinely different strategies for one of the hardest problems in this space — including being upfront about where they fell short."

**🎯 Delivery tip:** This is your strongest, most quotable sentence in the whole talk — the last one. Slow down, make eye contact, and let it land before moving to references.

---

## Slide 28 — References

**What to say:**
"These are the key papers this work is built on — the network architectures, the loss functions, the interpretability method, and the specific techniques behind each of the rare-lesion experiments. I'm happy to go into any of these in more detail during questions if it's useful."

**🎯 Delivery tip:** Keep this brief — ten seconds, tops. No one expects you to read a reference slide aloud; just acknowledge it exists and move on.

---

## Slide 29 — Thank You / Questions

**What to say:**
"Thank you all so much for your time and attention today. I'd love to take any questions you have — whether about a specific experiment, the limitations I've flagged, or where I think this should go next."

**🎯 Delivery tip:** Stop talking after this and *wait*. Silence right after "thank you" feels long to you, but it gives the room a natural moment to start asking questions instead of you having to fill the space.

---

## A Few Last Notes Before You Go In

- **If you forget a number mid-sentence**, don't panic and don't say "um, I forget." Just say "roughly X percent" and keep moving — nobody in the room has the exact numbers memorized either, and confidence matters far more than decimal precision.
- **If a professor pushes back on something**, resist the urge to defend immediately. Pause, nod, and say "that's a fair point" before responding — it never fails to de-escalate and makes you look thoughtful rather than defensive.
- **Your strongest moments in this whole talk** are slides 18, 21, and 25 — where you're honest about a limitation or a negative result. Don't rush through those. That's where you actually look like a real researcher, not just someone reading off results.
- **Breathe.** You built all of this. You already know it better than anyone else in that room ever will.
