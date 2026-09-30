// Teacher's view of one student: the same output panel, read-only.
import { OutputPanel } from './output.js';
import { toast } from './toast.js';

const root = document.getElementById('output');
new OutputPanel(root, { readonly: true, userId: +root.dataset.user, toast }).init();
