import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class LangGraphService {
  private apiUrl = 'http://localhost:8000/run';

  constructor(private http: HttpClient) {}

  callLangGraph(body: { message?: string; resume?: string; thread_id: string }): Observable<any> {
    return this.http.post(this.apiUrl, body);
  }
}
