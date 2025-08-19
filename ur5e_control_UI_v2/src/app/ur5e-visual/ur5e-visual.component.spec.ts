import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Ur5eVisualComponent } from './ur5e-visual.component';

describe('Ur5eVisualComponent', () => {
  let component: Ur5eVisualComponent;
  let fixture: ComponentFixture<Ur5eVisualComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ Ur5eVisualComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Ur5eVisualComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
